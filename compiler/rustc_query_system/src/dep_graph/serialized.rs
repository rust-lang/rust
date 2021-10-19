//! The data that we will serialize and deserialize.
//!
//! The dep-graph is serialized as a sequence of NodeInfo, with the dependencies
//! specified inline.  The total number of nodes and edges are stored as the last
//! 16 bytes of the file, so we can find them easily at decoding time.
//!
//! The serialisation is performed on-demand when each node is emitted. Using this
//! scheme, we do not need to keep the current graph in memory.
//!
//! The deserisalisation is performed manually, in order to convert from the stored
//! sequence of NodeInfos to the different arrays in SerializedDepGraph.  Since the
//! node and edge count are stored at the end of the file, all the arrays can be
//! pre-allocated with the right length.

use super::query::DepGraphQuery;
use super::{DepKind, DepNode, DepNodeIndex};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::Lock;
use rustc_index::vec::{Idx, IndexVec};
use rustc_serialize::opaque::{self, FileEncodeResult, FileEncoder, IntEncodedWithFixedSize};
use rustc_serialize::{Decodable, Decoder, Encodable};
use smallvec::SmallVec;
use std::convert::TryInto;

// The maximum value of `SerializedDepNodeIndex` leaves the upper two bits
// unused so that we can store multiple index types in `CompressedHybridIndex`,
// and use those bits to encode which index type it contains.
rustc_index::newtype_index! {
    pub struct SerializedDepNodeIndex {
        MAX = 0x7FFF_FFFF
    }
}

/// Data for use when recompiling the **current crate**.
#[derive(Debug)]
pub struct SerializedDepGraph<K: DepKind> {
    /// The set of all DepNodes in the graph
    nodes: IndexVec<SerializedDepNodeIndex, DepNode<K>>,
    /// The set of all Fingerprints in the graph. Each Fingerprint corresponds to
    /// the DepNode at the same index in the nodes vector.
    fingerprints: IndexVec<SerializedDepNodeIndex, Fingerprint>,
    /// For each DepNode, stores the list of edges originating from that
    /// DepNode. Encoded as a [start, end) pair indexing into edge_list_data,
    /// which holds the actual DepNodeIndices of the target nodes.
    edge_list_indices: IndexVec<SerializedDepNodeIndex, (u32, u32)>,
    /// A flattened list of all edge targets in the graph. Edge sources are
    /// implicit in edge_list_indices.
    edge_list_data: Vec<SerializedDepNodeIndex>,
    /// Reciprocal map to `nodes`.
    index: FxHashMap<DepNode<K>, SerializedDepNodeIndex>,
}

impl<K: DepKind> Default for SerializedDepGraph<K> {
    fn default() -> Self {
        SerializedDepGraph {
            nodes: Default::default(),
            fingerprints: Default::default(),
            edge_list_indices: Default::default(),
            edge_list_data: Default::default(),
            index: Default::default(),
        }
    }
}

impl<K: DepKind> SerializedDepGraph<K> {
    #[inline]
    pub fn edge_targets_from(&self, source: SerializedDepNodeIndex) -> &[SerializedDepNodeIndex] {
        let targets = self.edge_list_indices[source];
        &self.edge_list_data[targets.0 as usize..targets.1 as usize]
    }

    #[inline]
    pub fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> DepNode<K> {
        self.nodes[dep_node_index]
    }

    #[inline]
    pub fn node_to_index_opt(&self, dep_node: &DepNode<K>) -> Option<SerializedDepNodeIndex> {
        self.index.get(dep_node).cloned()
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node: &DepNode<K>) -> Option<Fingerprint> {
        self.index.get(dep_node).map(|&node_index| self.fingerprints[node_index])
    }

    #[inline]
    pub fn fingerprint_by_index(&self, dep_node_index: SerializedDepNodeIndex) -> Fingerprint {
        self.fingerprints[dep_node_index]
    }

    pub fn node_count(&self) -> usize {
        self.index.len()
    }
}

impl<'a, K: DepKind + Decodable<opaque::Decoder<'a>>> Decodable<opaque::Decoder<'a>>
    for SerializedDepGraph<K>
{
    #[instrument(level = "debug", skip(d))]
    fn decode(d: &mut opaque::Decoder<'a>) -> Result<SerializedDepGraph<K>, String> {
        let start_position = d.position();

        // The last 16 bytes are the node count and edge count.
        debug!("position: {:?}", d.position());
        d.set_position(d.data.len() - 2 * IntEncodedWithFixedSize::ENCODED_SIZE);
        debug!("position: {:?}", d.position());

        let node_count = IntEncodedWithFixedSize::decode(d)?.0 as usize;
        let edge_count = IntEncodedWithFixedSize::decode(d)?.0 as usize;
        debug!(?node_count, ?edge_count);

        debug!("position: {:?}", d.position());
        d.set_position(start_position);
        debug!("position: {:?}", d.position());

        let mut nodes = IndexVec::with_capacity(node_count);
        let mut fingerprints = IndexVec::with_capacity(node_count);
        let mut edge_list_indices = IndexVec::with_capacity(node_count);
        let mut edge_list_data = Vec::with_capacity(edge_count);

        for _index in 0..node_count {
            d.read_struct(|d| {
                let dep_node: DepNode<K> = d.read_struct_field("node", Decodable::decode)?;
                let _i: SerializedDepNodeIndex = nodes.push(dep_node);
                debug_assert_eq!(_i.index(), _index);

                let fingerprint: Fingerprint =
                    d.read_struct_field("fingerprint", Decodable::decode)?;
                let _i: SerializedDepNodeIndex = fingerprints.push(fingerprint);
                debug_assert_eq!(_i.index(), _index);

                d.read_struct_field("edges", |d| {
                    d.read_seq(|d, len| {
                        let start = edge_list_data.len().try_into().unwrap();
                        for _ in 0..len {
                            let edge = d.read_seq_elt(Decodable::decode)?;
                            edge_list_data.push(edge);
                        }
                        let end = edge_list_data.len().try_into().unwrap();
                        let _i: SerializedDepNodeIndex = edge_list_indices.push((start, end));
                        debug_assert_eq!(_i.index(), _index);
                        Ok(())
                    })
                })
            })?;
        }

        let index: FxHashMap<_, _> =
            nodes.iter_enumerated().map(|(idx, &dep_node)| (dep_node, idx)).collect();

        Ok(SerializedDepGraph { nodes, fingerprints, edge_list_indices, edge_list_data, index })
    }
}

#[derive(Debug, Encodable, Decodable)]
pub struct NodeInfo<K: DepKind> {
    node: DepNode<K>,
    fingerprint: Fingerprint,
    edges: SmallVec<[DepNodeIndex; 8]>,
}

struct Stat<K: DepKind> {
    kind: K,
    node_counter: u64,
    edge_counter: u64,
}

struct EncoderState<K: DepKind> {
    encoder: FileEncoder,
    total_node_count: usize,
    total_edge_count: usize,
    result: FileEncodeResult,
    stats: Option<FxHashMap<K, Stat<K>>>,
}

impl<K: DepKind> EncoderState<K> {
    fn new(encoder: FileEncoder, record_stats: bool) -> Self {
        Self {
            encoder,
            total_edge_count: 0,
            total_node_count: 0,
            result: Ok(()),
            stats: if record_stats { Some(FxHashMap::default()) } else { None },
        }
    }

    #[instrument(level = "debug", skip(self, record_graph))]
    fn encode_node(
        &mut self,
        node: &NodeInfo<K>,
        record_graph: &Option<Lock<DepGraphQuery<K>>>,
    ) -> DepNodeIndex {
        let index = DepNodeIndex::new(self.total_node_count);
        self.total_node_count += 1;

        let edge_count = node.edges.len();
        self.total_edge_count += edge_count;

        if let Some(record_graph) = &record_graph {
            // Do not ICE when a query is called from within `with_query`.
            if let Some(record_graph) = &mut record_graph.try_lock() {
                record_graph.push(index, node.node, &node.edges);
            }
        }

        if let Some(stats) = &mut self.stats {
            let kind = node.node.kind;

            let stat = stats.entry(kind).or_insert(Stat { kind, node_counter: 0, edge_counter: 0 });
            stat.node_counter += 1;
            stat.edge_counter += edge_count as u64;
        }

        debug!(?index, ?node);
        let encoder = &mut self.encoder;
        if self.result.is_ok() {
            self.result = node.encode(encoder);
        }
        index
    }

    fn finish(self, profiler: &SelfProfilerRef) -> FileEncodeResult {
        let Self { mut encoder, total_node_count, total_edge_count, result, stats: _ } = self;
        let () = result?;

        let node_count = total_node_count.try_into().unwrap();
        let edge_count = total_edge_count.try_into().unwrap();

        debug!(?node_count, ?edge_count);
        debug!("position: {:?}", encoder.position());
        IntEncodedWithFixedSize(node_count).encode(&mut encoder)?;
        IntEncodedWithFixedSize(edge_count).encode(&mut encoder)?;
        debug!("position: {:?}", encoder.position());
        // Drop the encoder so that nothing is written after the counts.
        let result = encoder.flush();
        // FIXME(rylev): we hardcode the dep graph file name so we don't need a dependency on
        // rustc_incremental just for that.
        profiler.artifact_size("dep_graph", "dep-graph.bin", encoder.position() as u64);
        result
    }
}

pub struct GraphEncoder<K: DepKind> {
    status: Lock<EncoderState<K>>,
    record_graph: Option<Lock<DepGraphQuery<K>>>,
}

impl<K: DepKind + Encodable<FileEncoder>> GraphEncoder<K> {
    pub fn new(
        encoder: FileEncoder,
        prev_node_count: usize,
        record_graph: bool,
        record_stats: bool,
    ) -> Self {
        let record_graph =
            if record_graph { Some(Lock::new(DepGraphQuery::new(prev_node_count))) } else { None };
        let status = Lock::new(EncoderState::new(encoder, record_stats));
        GraphEncoder { status, record_graph }
    }

    pub(crate) fn with_query(&self, f: impl Fn(&DepGraphQuery<K>)) {
        if let Some(record_graph) = &self.record_graph {
            f(&record_graph.lock())
        }
    }

    pub(crate) fn print_incremental_info(
        &self,
        total_read_count: u64,
        total_duplicate_read_count: u64,
    ) {
        let status = self.status.lock();
        if let Some(record_stats) = &status.stats {
            let mut stats: Vec<_> = record_stats.values().collect();
            stats.sort_by_key(|s| -(s.node_counter as i64));

            const SEPARATOR: &str = "[incremental] --------------------------------\
                                     ----------------------------------------------\
                                     ------------";

            eprintln!("[incremental]");
            eprintln!("[incremental] DepGraph Statistics");
            eprintln!("{}", SEPARATOR);
            eprintln!("[incremental]");
            eprintln!("[incremental] Total Node Count: {}", status.total_node_count);
            eprintln!("[incremental] Total Edge Count: {}", status.total_edge_count);

            if cfg!(debug_assertions) {
                eprintln!("[incremental] Total Edge Reads: {}", total_read_count);
                eprintln!(
                    "[incremental] Total Duplicate Edge Reads: {}",
                    total_duplicate_read_count
                );
            }

            eprintln!("[incremental]");
            eprintln!(
                "[incremental]  {:<36}| {:<17}| {:<12}| {:<17}|",
                "Node Kind", "Node Frequency", "Node Count", "Avg. Edge Count"
            );
            eprintln!("{}", SEPARATOR);

            for stat in stats {
                let node_kind_ratio =
                    (100.0 * (stat.node_counter as f64)) / (status.total_node_count as f64);
                let node_kind_avg_edges = (stat.edge_counter as f64) / (stat.node_counter as f64);

                eprintln!(
                    "[incremental]  {:<36}|{:>16.1}% |{:>12} |{:>17.1} |",
                    format!("{:?}", stat.kind),
                    node_kind_ratio,
                    stat.node_counter,
                    node_kind_avg_edges,
                );
            }

            eprintln!("{}", SEPARATOR);
            eprintln!("[incremental]");
        }
    }

    pub(crate) fn send(
        &self,
        profiler: &SelfProfilerRef,
        node: DepNode<K>,
        fingerprint: Fingerprint,
        edges: SmallVec<[DepNodeIndex; 8]>,
    ) -> DepNodeIndex {
        let _prof_timer = profiler.generic_activity("incr_comp_encode_dep_graph");
        let node = NodeInfo { node, fingerprint, edges };
        self.status.lock().encode_node(&node, &self.record_graph)
    }

    pub fn finish(self, profiler: &SelfProfilerRef) -> FileEncodeResult {
        let _prof_timer = profiler.generic_activity("incr_comp_encode_dep_graph");
        self.status.into_inner().finish(profiler)
    }
}
