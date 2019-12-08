# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/pairwise_metric'

module Rumale
  module NearestNeighbors
    # BallTree is a class that implements the nearest neigbor searcher based on ball tree.
    #
    # *Reference*
    class KMeansTree
      attr_reader :tree

      def initialize(data: nil, leaf_size: 1)
        @params = {}
        @params[:leaf_size] = leaf_size
        @data = data
        @tree = build_tree(Numo::Int32.cast([*0...@data.shape[0]]))
      end

      def query(q, k = 10)
        rel_node = search(q, @tree, k)
        dist_arr = calc_distances(q, @data[rel_node.sample_ids, true])
        rank_ids = dist_arr.sort_index[0...k]
        [rel_node.sample_ids[rank_ids].dup, dist_arr[rank_ids].dup]
      end

      private

      Node = Struct.new(:sample_ids, :n_samples, :project_vec, :threshold, :left, :right)

      private_constant :Node

      def search(q, node, k)
        return node if node.project_vec.nil?

        dist = node.project_vec.dot(q)

        if dist < node.threshold
          if node.left.n_samples < k
            node
          else
            search(q, node.left, k)
          end
        else
          if node.right.n_samples < k
            node
          else
            search(q, node.right, k)
          end
        end
      end

      def build_tree(sample_ids)
        n_samples = sample_ids.size
        node = Node.new
        node.n_samples = n_samples
        node.sample_ids = sample_ids
        return node if n_samples <= @params[:leaf_size]

        samples = @data[sample_ids, true]
        centroids = update_centroids(samples, init_centroids(samples))

        node.project_vec = centroids[0, true] - centroids[1, true]
        line = samples.dot(node.project_vec)
        node.threshold = line.median

        node.left = build_tree(sample_ids[line.lt(node.threshold)])
        node.right = build_tree(sample_ids[line.ge(node.threshold)])
        node
      end

      def calc_distances(query, samples)
        Rumale::PairwiseMetric.euclidean_distance(query.expand_dims(0), samples).flatten.dup
      end

      def init_centroids(samples)
        centroid = samples.mean(0)
        left = samples[calc_distances(centroid, samples).max_index, true]
        right = samples[calc_distances(left, samples).max_index, true]
        Numo::NArray.vstack([left, right])
      end

      def update_centroids(samples, centroids, n_iters = 10)
        n_centroids = centroids.shape[0]
        n_iters.times do
          dist_mat = Rumale::PairwiseMetric.squared_error(samples, centroids)
          clstr_lbl = dist_mat.min_index(axis: 1) - Numo::Int32[*0.step(dist_mat.size - 1, n_centroids)]
          n_centroids.times do |n|
            centroids[n, true] = samples[clstr_lbl.eq(n).where, true].mean(axis: 0)
          end
        end
        centroids
      end
    end
  end
end
