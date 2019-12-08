# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::KMeansTree do
  let(:three_clusters) { three_clusters_dataset }
  let(:x) { three_clusters[0] }
  let(:y) { three_clusters[1] }
  let(:tree) { described_class.new(data: x, leaf_size: 50) }

  it 'search' do
    pp tree.query(x[0,true], 10)
    puts '---'
    dist_arr= Rumale::PairwiseMetric.euclidean_distance(x[0, true].expand_dims(0), x)
    rank_ids = dist_arr.sort_index[0...10]
    pp rank_ids
    pp dist_arr[rank_ids]
  end
end
