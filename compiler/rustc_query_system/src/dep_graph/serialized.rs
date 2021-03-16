//! The data that we will serialize and deserialize.

use super::query::DepGraphQuery;
use super::{DepKind, DepNode, DepNodeIndex};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::{Lock, Lrc};
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
    pub nodes: IndexVec<SerializedDepNodeIndex, DepNode<K>>,
    /// The set of all Fingerprints in the graph. Each Fingerprint corresponds to
    /// the DepNode at the same index in the nodes vector.
    pub fingerprints: IndexVec<SerializedDepNodeIndex, Fingerprint>,
    /// For each DepNode, stores the list of edges originating from that
    /// DepNode. Encoded as a [start, end) pair indexing into edge_list_data,
    /// which holds the actual DepNodeIndices of the target nodes.
    pub edge_list_indices: IndexVec<SerializedDepNodeIndex, (u32, u32)>,
    /// A flattened list of all edge targets in the graph. Edge sources are
    /// implicit in edge_list_indices.
    pub edge_list_data: Vec<SerializedDepNodeIndex>,
}

impl<K: DepKind> Default for SerializedDepGraph<K> {
    fn default() -> Self {
        SerializedDepGraph {
            nodes: Default::default(),
            fingerprints: Default::default(),
            edge_list_indices: Default::default(),
            edge_list_data: Default::default(),
        }
    }
}

impl<K: DepKind> SerializedDepGraph<K> {
    #[inline]
    pub fn edge_targets_from(&self, source: SerializedDepNodeIndex) -> &[SerializedDepNodeIndex] {
        let targets = self.edge_list_indices[source];
        &self.edge_list_data[targets.0 as usize..targets.1 as usize]
    }
}

impl<'a, K: DepKind + Decodable<opaque::Decoder<'a>>> Decodable<opaque::Decoder<'a>>
    for SerializedDepGraph<K>
{
    #[instrument(skip(d))]
    fn decode(d: &mut opaque::Decoder<'a>) -> Result<SerializedDepGraph<K>, String> {
        let position = d.position();

        // The last 16 bytes are the node count and edge count.
        debug!("position: {:?}", d.position());
        d.set_position(d.data.len() - 2 * IntEncodedWithFixedSize::ENCODED_SIZE);
        debug!("position: {:?}", d.position());

        let node_count = IntEncodedWithFixedSize::decode(d)?.0 as usize;
        let edge_count = IntEncodedWithFixedSize::decode(d)?.0 as usize;
        debug!(?node_count, ?edge_count);

        debug!("position: {:?}", d.position());
        d.set_position(position);
        debug!("position: {:?}", d.position());

        let mut nodes = IndexVec::with_capacity(node_count);
        let mut fingerprints = IndexVec::with_capacity(node_count);
        let mut edge_list_indices = IndexVec::with_capacity(node_count);
        let mut edge_list_data = Vec::with_capacity(edge_count);

        for _index in 0..node_count {
            d.read_struct("NodeInfo", 3, |d| {
                let dep_node: DepNode<K> = d.read_struct_field("node", 0, Decodable::decode)?;
                let _i: SerializedDepNodeIndex = nodes.push(dep_node);
                debug_assert_eq!(_i.index(), _index);

                let fingerprint: Fingerprint =
                    d.read_struct_field("fingerprint", 1, Decodable::decode)?;
                let _i: SerializedDepNodeIndex = fingerprints.push(fingerprint);
                debug_assert_eq!(_i.index(), _index);

                d.read_struct_field("edges", 2, |d| {
                    d.read_seq(|d, len| {
                        let start = edge_list_data.len().try_into().unwrap();
                        for e in 0..len {
                            let edge = d.read_seq_elt(e, Decodable::decode)?;
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

        Ok(SerializedDepGraph { nodes, fingerprints, edge_list_indices, edge_list_data })
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

struct Stats<K: DepKind> {
    stats: FxHashMap<K, Stat<K>>,
    total_node_count: usize,
    total_edge_count: usize,
}

#[instrument(skip(encoder, _record_graph, record_stats))]
fn encode_node<K: DepKind>(
    encoder: &mut FileEncoder,
    _index: DepNodeIndex,
    node: &NodeInfo<K>,
    _record_graph: &Option<Lrc<Lock<DepGraphQuery<K>>>>,
    record_stats: &Option<Lrc<Lock<Stats<K>>>>,
) -> FileEncodeResult {
    #[cfg(debug_assertions)]
    if let Some(record_graph) = &_record_graph {
        // Do not ICE when a query is called from within `with_query`.
        if let Some(record_graph) = &mut record_graph.try_lock() {
            record_graph.push(_index, node.node, &node.edges);
        }
    }

    if let Some(record_stats) = &record_stats {
        let mut stats = record_stats.lock();
        let kind = node.node.kind;
        let edge_count = node.edges.len();

        let stat =
            stats.stats.entry(kind).or_insert(Stat { kind, node_counter: 0, edge_counter: 0 });
        stat.node_counter += 1;
        stat.edge_counter += edge_count as u64;
        stats.total_node_count += 1;
        stats.total_edge_count += edge_count;
    }

    debug!(?_index, ?node);
    node.encode(encoder)
}

fn encode_counts(
    mut encoder: FileEncoder,
    node_count: usize,
    edge_count: usize,
) -> FileEncodeResult {
    let node_count = node_count.try_into().unwrap();
    let edge_count = edge_count.try_into().unwrap();

    debug!(?node_count, ?edge_count);
    debug!("position: {:?}", encoder.position());
    IntEncodedWithFixedSize(node_count).encode(&mut encoder)?;
    IntEncodedWithFixedSize(edge_count).encode(&mut encoder)?;
    debug!("position: {:?}", encoder.position());
    // Drop the encoder so that nothing is written after the counts.
    encoder.flush()
}

pub struct GraphEncoder<K: DepKind> {
    status: Lock<(FileEncoder, DepNodeIndex, usize, FileEncodeResult)>,
    record_graph: Option<Lrc<Lock<DepGraphQuery<K>>>>,
    record_stats: Option<Lrc<Lock<Stats<K>>>>,
}

impl<K: DepKind + Encodable<FileEncoder>> GraphEncoder<K> {
    pub fn new(
        encoder: FileEncoder,
        prev_node_count: usize,
        record_graph: bool,
        record_stats: bool,
    ) -> Self {
        let record_graph = if cfg!(debug_assertions) && record_graph {
            Some(Lrc::new(Lock::new(DepGraphQuery::new(prev_node_count))))
        } else {
            None
        };
        let record_stats = if record_stats {
            Some(Lrc::new(Lock::new(Stats {
                stats: FxHashMap::default(),
                total_node_count: 0,
                total_edge_count: 0,
            })))
        } else {
            None
        };
        let status = Lock::new((encoder, DepNodeIndex::new(0), 0, Ok(())));
        GraphEncoder { status, record_graph, record_stats }
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
        if let Some(record_stats) = &self.record_stats {
            let record_stats = record_stats.lock();

            let mut stats: Vec<_> = record_stats.stats.values().collect();
            stats.sort_by_key(|s| -(s.node_counter as i64));

            const SEPARATOR: &str = "[incremental] --------------------------------\
                                     ----------------------------------------------\
                                     ------------";

            eprintln!("[incremental]");
            eprintln!("[incremental] DepGraph Statistics");
            eprintln!("{}", SEPARATOR);
            eprintln!("[incremental]");
            eprintln!("[incremental] Total Node Count: {}", record_stats.total_node_count);
            eprintln!("[incremental] Total Edge Count: {}", record_stats.total_edge_count);

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
                    (100.0 * (stat.node_counter as f64)) / (record_stats.total_node_count as f64);
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
        node: DepNode<K>,
        fingerprint: Fingerprint,
        edges: SmallVec<[DepNodeIndex; 8]>,
    ) -> DepNodeIndex {
        let &mut (ref mut encoder, ref mut next_index, ref mut edge_count, ref mut result) =
            &mut *self.status.lock();
        let index = next_index.clone();
        next_index.increment_by(1);
        *edge_count += edges.len();
        *result = std::mem::replace(result, Ok(())).and_then(|()| {
            let node = NodeInfo { node, fingerprint, edges };
            encode_node(encoder, index, &node, &self.record_graph, &self.record_stats)
        });
        index
    }

    pub fn finish(self) -> FileEncodeResult {
        let (encoder, node_count, edge_count, result) = self.status.into_inner();
        let () = result?;
        let node_count = node_count.index();

        encode_counts(encoder, node_count, edge_count)
    }
}
