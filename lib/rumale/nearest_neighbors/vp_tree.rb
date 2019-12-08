# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/pairwise_metric'

module Rumale
  module NearestNeighbors
    # VPTree is a class that implements the nearest neigbor searcher based on vantage point tree.
    #
    # *Reference*
    # P N. Yianilos, "Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces," Proc. SODA'93, pp. 311--321, 1993.
    class VPTree
      def initialize(x, min_samples: 1, n_rand_samples: 50, random_seed: nil)
        @params = {}
        @params[:min_samples] = min_samples
        @params[:n_rand_samples] = n_rand_samples
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @data = x
        @tree = build_tree(Numo::Int32.cast([*0...@data.shape[0]]))
      end

      def query(q, k = 10)
        rel_node = search(q, @tree, k)
        dist_arr = calc_distances(q, @data[rel_node.sample_ids, true])
        rank_ids = dist_arr.sort_index[0...k]
        [rel_node.sample_ids[rank_ids].dup, dist_arr[rank_ids].dup]
      end

      private

      Node = Struct.new(:sample_ids, :n_samples, :vantage_point_id, :threshold, :left, :right)

      private_constant :Node

      def search(q, node, k)
        return node if node.vantage_point_id.nil?

        dist = Math.sqrt(((q - @data[node.vantage_point_id, true])**2).sum)

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
        return node if n_samples <= @params[:min_samples]

        node.vantage_point_id = select_vantage_point_id(sample_ids)
        distance_arr = calc_distances(@data[node.vantage_point_id, true], @data[sample_ids, true])
        node.threshold = distance_arr.median

        node.left = build_tree(sample_ids[distance_arr.lt(node.threshold)])
        node.right = build_tree(sample_ids[distance_arr.ge(node.threshold)])
        node
      end

      def select_vantage_point_id(sample_ids)
        n_samples = sample_ids.size
        n_r_samples = [n_samples, @params[:n_rand_samples]].min
        pivot_ids = random_sample(sample_ids, n_r_samples)
        best_pid = pivot_ids[0]
        best_var = 0.0
        pivot_ids.each do |pid|
          #target_ids = random_sample(sample_ids, n_r_samples)
          #distance_arr = calc_distances(@data[pid, true], @data[target_ids, true])
          distance_arr = calc_distances(@data[pid, true], @data[sample_ids, true])
          mu = distance_arr.median
          curr_var = ((distance_arr - mu)**2).mean
          if curr_var > best_var
            best_var = curr_var
            best_pid = pid
          end
        end
        best_pid
      end

      def calc_distances(query, samples)
        Rumale::PairwiseMetric.euclidean_distance(query.expand_dims(0), samples).flatten.dup
      end

      def random_sample(sample_ids, n_rand_samples)
        n_samples = sample_ids.size
        rand_select = [*0...n_samples].shuffle(random: @rng).shift(n_rand_samples)
        sample_ids[rand_select]
      end
    end
  end
end
