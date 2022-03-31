//! The data that we will serialize and deserialize.
//!
//! The dep-graph is serialized as a sequence of NodeInfo, with the dependencies
//! specified inline.  The total number of nodes and edges are stored as the last
//! 16 bytes of the file, so we can find them easily at decoding time.
//!
//! The serialisation is performed on-demand when each node is emitted. Using this
//! scheme, we do not need to keep the current graph in memory.
//!
//! The deserialization is performed manually, in order to convert from the stored
//! sequence of NodeInfos to the different arrays in SerializedDepGraph.  Since the
//! node and edge count are stored at the end of the file, all the arrays can be
//! pre-allocated with the right length.

use super::query::DepGraphQuery;
use super::{DepKind, DepNode, DepNodeIndex};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::owning_ref::OwningRef;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::Lock;
use rustc_index::vec::IndexVec;
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder, IntEncodedWithFixedSize, MemDecoder};
use rustc_serialize::{Decodable, Encodable};
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
///
/// The DepGraph is backed on-disk and read on-demand through a Mmap.
/// The file layout is as follows.
///
/// The DepGraph starts with the version header, handled by rustc_incremental.
/// It is followed by the concatenation of all node dependency information as:
/// - the query result fingerprint (size_of<Fingerprint> bytes)
/// - the number of dependency edges (4 bytes)
/// - the dependencies indices (array of 4-byte integers)
///
/// Finding this information inside the file is handled by a "glossary" written at the end.
/// This glossary is an array of `(DepNode, u32)` pairs.  This array is used to make the
/// correspondence between the `SerializedDepNodeIndex` (ie. the index into this array),
/// and the `DepNode`.  The `u32` is the position of the dependency information (Fingerprint +
/// array of dependencies) inside the file.  The glossary array is directly mmapped into `nodes`.
///
/// The file finished with two `u64`, which are the number of entries in the glossary
/// and its position in the file.
///
/// Graphically, we have:
///          beginning of nodes                        beginning of glossary ---------------+
///          v                                         v                                    |
/// --------------------------------------------------------------------------------------------
/// | HEADER | ... | Fingerprint | Length | Deps | ... | ... | DepNode | u32 | ... | u64 | u64 |
/// --------------------------------------------------------------------------------------------
///                ^                                                      |         node
///                start for node i --------------------------------------+         count
///
/// In order to recover an index from a DepNode, we populate a hash-map in `index`.
pub struct SerializedDepGraph<K: DepKind> {
    /// The set of all DepNodes in the graph and their position in the mmap.
    nodes: Option<OwningRef<Mmap, [(DepNode<K>, u32)]>>,
    /// Reciprocal map to `nodes`.
    index: FxHashMap<DepNode<K>, SerializedDepNodeIndex>,
}

impl<K: DepKind> Default for SerializedDepGraph<K> {
    fn default() -> Self {
        SerializedDepGraph { nodes: None, index: Default::default() }
    }
}

impl<K: DepKind> SerializedDepGraph<K> {
    #[inline]
    fn decoder_at(&self, dep_node_index: SerializedDepNodeIndex) -> Option<MemDecoder<'_>> {
        let nodes = self.nodes.as_ref()?;
        let dep_node_index = dep_node_index.as_usize();
        let position = nodes[dep_node_index].1 as usize;
        let data = &nodes.owner()[position..];
        let decoder = MemDecoder::new(data, 0);
        Some(decoder)
    }

    #[inline]
    pub fn node_to_index_opt(&self, dep_node: &DepNode<K>) -> Option<SerializedDepNodeIndex> {
        self.index.get(dep_node).cloned()
    }

    #[inline]
    pub fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> DepNode<K> {
        let dep_node_index = dep_node_index.as_usize();
        self.nodes.as_ref().unwrap()[dep_node_index].0
    }

    #[inline]
    pub fn fingerprint_by_index(&self, dep_node_index: SerializedDepNodeIndex) -> Fingerprint {
        if let Some(decoder) = self.decoder_at(dep_node_index) {
            let &fingerprint = unsafe { decoder.mmap_at::<Fingerprint>(0) };
            fingerprint
        } else {
            Fingerprint::ZERO
        }
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node: &DepNode<K>) -> Option<Fingerprint> {
        let index = self.index.get(dep_node).cloned()?;
        Some(self.fingerprint_by_index(index))
    }

    #[inline]
    pub fn edge_targets_from(&self, source: SerializedDepNodeIndex) -> &[SerializedDepNodeIndex] {
        // The encoder has checked that there is no padding there.
        if let Some(decoder) = self.decoder_at(source) {
            let position = std::mem::size_of::<Fingerprint>();
            let &length = unsafe { decoder.mmap_at::<u32>(position) };
            unsafe {
                decoder.mmap_slice_at::<SerializedDepNodeIndex>(position + 4, length as usize)
            }
        } else {
            &[]
        }
    }

    pub fn node_count(&self) -> usize {
        self.index.len()
    }

    #[instrument(level = "debug", skip(mmap))]
    pub fn decode(mmap: Mmap) -> SerializedDepGraph<K> {
        let data = mmap.as_ref();

        // The last 16 bytes are the node count, edge count and nodes position.
        let start_position = data.len() - 2 * IntEncodedWithFixedSize::ENCODED_SIZE;
        let mut d = MemDecoder::new(data, start_position);
        debug!("position: {:?}", d.position());

        let node_count = IntEncodedWithFixedSize::decode(&mut d).0 as usize;
        let nodes_position = IntEncodedWithFixedSize::decode(&mut d).0 as usize;
        debug!(?node_count, ?nodes_position);

        let nodes = OwningRef::new(mmap).map(|mmap| {
            let d = MemDecoder::new(mmap, nodes_position);
            unsafe { d.mmap_slice_at::<(DepNode<K>, u32)>(nodes_position, node_count) }
        });

        let index: FxHashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(idx, &(dep_node, _))| (dep_node, SerializedDepNodeIndex::from_usize(idx)))
            .collect();

        SerializedDepGraph { nodes: Some(nodes), index }
    }
}

struct EncoderState<K: DepKind> {
    encoder: FileEncoder,
    total_edge_count: usize,
    nodes: IndexVec<DepNodeIndex, (DepNode<K>, u32)>,
    stats: Option<FxHashMap<K, Stat<K>>>,
}

impl<K: DepKind> EncoderState<K> {
    fn new(encoder: FileEncoder, record_stats: bool) -> Self {
        Self {
            encoder,
            total_edge_count: 0,
            nodes: IndexVec::default(),
            stats: record_stats.then(FxHashMap::default),
        }
    }

    fn try_encode_node(&mut self, node: &NodeInfo<K>) -> usize {
        let encoder = &mut self.encoder;
        let start_pos = encoder.write_mmap(&node.fingerprint);
        let _pos = encoder.write_mmap::<u32>(&node.edges.len().try_into().unwrap());
        debug_assert_eq!(_pos, start_pos + std::mem::size_of::<Fingerprint>());
        let _pos = encoder.write_mmap_slice::<DepNodeIndex>(&node.edges[..]);
        debug_assert_eq!(_pos, start_pos + std::mem::size_of::<Fingerprint>() + 4);
        start_pos
    }

    fn encode_node(
        &mut self,
        node: &NodeInfo<K>,
        record_graph: &Option<Lock<DepGraphQuery<K>>>,
    ) -> DepNodeIndex {
        let edge_count = node.edges.len();
        self.total_edge_count += edge_count;

        let position = self.try_encode_node(node);
        debug_assert!(position & (std::mem::align_of::<Fingerprint>() - 1) == 0);
        debug!(?position);

        let index = self.nodes.push((node.node, position.try_into().unwrap()));

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

        index
    }

    fn finish(self, profiler: &SelfProfilerRef) -> FileEncodeResult {
        let Self { mut encoder, nodes, total_edge_count: _, stats: _ } = self;

        let node_count = nodes.len().try_into().unwrap();
        let nodes_position = encoder.write_mmap_slice(&nodes.raw[..]);
        let nodes_position = nodes_position.try_into().unwrap();

        debug!(?node_count, ?nodes_position);
        debug!("position: {:?}", encoder.position());
        IntEncodedWithFixedSize(node_count).encode(&mut encoder);
        IntEncodedWithFixedSize(nodes_position).encode(&mut encoder);
        debug!("position: {:?}", encoder.position());
        // Drop the encoder so that nothing is written after the counts.
        let result = encoder.finish();
        if let Ok(position) = result {
            // FIXME(rylev): we hardcode the dep graph file name so we
            // don't need a dependency on rustc_incremental just for that.
            profiler.artifact_size("dep_graph", "dep-graph.bin", position as u64);
        }
        result
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

pub struct GraphEncoder<K: DepKind> {
    status: Lock<EncoderState<K>>,
    record_graph: Option<Lock<DepGraphQuery<K>>>,
}

impl<K: DepKind> GraphEncoder<K> {
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

            let total_node_count = status.nodes.len();

            eprintln!("[incremental]");
            eprintln!("[incremental] DepGraph Statistics");
            eprintln!("{}", SEPARATOR);
            eprintln!("[incremental]");
            eprintln!("[incremental] Total Node Count: {}", total_node_count);
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
                    (100.0 * (stat.node_counter as f64)) / (total_node_count as f64);
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
