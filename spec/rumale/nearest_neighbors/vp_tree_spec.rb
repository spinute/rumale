# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::VPTree do
  let(:three_clusters) { three_clusters_dataset }
  let(:x) { three_clusters[0] }
  let(:y) { three_clusters[1] }
  let(:vp_tree) { described_class.new(x, n_rand_samples: 100) }

  it 'search' do
    pp vp_tree.query(x[0, true], 10)
  end
end
