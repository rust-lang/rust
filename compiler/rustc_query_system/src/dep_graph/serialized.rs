//! The data that we will serialize and deserialize.
//!
//! Notionally, the dep-graph is a sequence of NodeInfo with the dependencies
//! specified inline. The total number of nodes and edges are stored as the last
//! 16 bytes of the file, so we can find them easily at decoding time.
//!
//! The serialisation is performed on-demand when each node is emitted. Using this
//! scheme, we do not need to keep the current graph in memory.
//!
//! The deserialization is performed manually, in order to convert from the stored
//! sequence of NodeInfos to the different arrays in SerializedDepGraph. Since the
//! node and edge count are stored at the end of the file, all the arrays can be
//! pre-allocated with the right length.
//!
//! The encoding of the de-pgraph is generally designed around the fact that fixed-size
//! reads of encoded data are generally faster than variable-sized reads. Ergo we adopt
//! essentially the same varint encoding scheme used in the rmeta format; the edge lists
//! for each node on the graph store a 2-bit integer which is the number of bytes per edge
//! index in that node's edge list. We effectively ignore that an edge index of 0 could be
//! encoded with 0 bytes in order to not require 3 bits to store the byte width of the edges.
//! The overhead of calculating the correct byte width for each edge is mitigated by
//! building edge lists with [`EdgesVec`] which keeps a running max of the edges in a node.
//!
//! When we decode this data, we do not immediately create [`SerializedDepNodeIndex`] and
//! instead keep the data in its denser serialized form which lets us turn our on-disk size
//! efficiency directly into a peak memory reduction. When we convert these encoded-in-memory
//! values into their fully-deserialized type, we use a fixed-size read of the encoded array
//! then mask off any errant bytes we read. The array of edge index bytes is padded to permit this.
//!
//! We also encode and decode the entire rest of each node using [`SerializedNodeHeader`]
//! to let this encoding and decoding be done in one fixed-size operation. These headers contain
//! two [`Fingerprint`]s along with the serialized [`DepKind`], and the number of edge indices
//! in the node and the number of bytes used to encode the edge indices for this node. The
//! [`DepKind`], number of edges, and bytes per edge are all bit-packed together, if they fit.
//! If the number of edges in this node does not fit in the bits available in the header, we
//! store it directly after the header with leb128.

use std::iter;
use std::marker::PhantomData;
use std::sync::Arc;

use rustc_data_structures::fingerprint::{Fingerprint, PackedFingerprint};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::outline;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::unhash::UnhashMap;
use rustc_index::{Idx, IndexVec};
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder, IntEncodedWithFixedSize, MemDecoder};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use tracing::{debug, instrument};

use super::graph::{DepNodeColor, DepNodeColorMap};
use super::query::DepGraphQuery;
use super::{DepKind, DepNode, DepNodeIndex, Deps};
use crate::dep_graph::edges::EdgesVec;

// The maximum value of `SerializedDepNodeIndex` leaves the upper two bits
// unused so that we can store multiple index types in `CompressedHybridIndex`,
// and use those bits to encode which index type it contains.
rustc_index::newtype_index! {
    #[encodable]
    #[max = 0x7FFF_FFFF]
    pub struct SerializedDepNodeIndex {}
}

const DEP_NODE_SIZE: usize = size_of::<SerializedDepNodeIndex>();
/// Amount of padding we need to add to the edge list data so that we can retrieve every
/// SerializedDepNodeIndex with a fixed-size read then mask.
const DEP_NODE_PAD: usize = DEP_NODE_SIZE - 1;
/// Number of bits we need to store the number of used bytes in a SerializedDepNodeIndex.
/// Note that wherever we encode byte widths like this we actually store the number of bytes used
/// minus 1; for a 4-byte value we technically would have 5 widths to store, but using one byte to
/// store zeroes (which are relatively rare) is a decent tradeoff to save a bit in our bitfields.
const DEP_NODE_WIDTH_BITS: usize = DEP_NODE_SIZE / 2;

/// Data for use when recompiling the **current crate**.
#[derive(Debug, Default)]
pub struct SerializedDepGraph {
    /// The set of all DepNodes in the graph
    nodes: IndexVec<SerializedDepNodeIndex, DepNode>,
    /// The set of all Fingerprints in the graph. Each Fingerprint corresponds to
    /// the DepNode at the same index in the nodes vector.
    fingerprints: IndexVec<SerializedDepNodeIndex, Fingerprint>,
    /// For each DepNode, stores the list of edges originating from that
    /// DepNode. Encoded as a [start, end) pair indexing into edge_list_data,
    /// which holds the actual DepNodeIndices of the target nodes.
    edge_list_indices: IndexVec<SerializedDepNodeIndex, EdgeHeader>,
    /// A flattened list of all edge targets in the graph, stored in the same
    /// varint encoding that we use on disk. Edge sources are implicit in edge_list_indices.
    edge_list_data: Vec<u8>,
    /// Stores a map from fingerprints to nodes per dep node kind.
    /// This is the reciprocal of `nodes`.
    index: Vec<UnhashMap<PackedFingerprint, SerializedDepNodeIndex>>,
}

impl SerializedDepGraph {
    #[inline]
    pub fn edge_targets_from(
        &self,
        source: SerializedDepNodeIndex,
    ) -> impl Iterator<Item = SerializedDepNodeIndex> + Clone {
        let header = self.edge_list_indices[source];
        let mut raw = &self.edge_list_data[header.start()..];
        // Figure out where the edge list for `source` ends by getting the start index of the next
        // edge list, or the end of the array if this is the last edge.
        let end = self
            .edge_list_indices
            .get(source + 1)
            .map(|h| h.start())
            .unwrap_or_else(|| self.edge_list_data.len() - DEP_NODE_PAD);

        // The number of edges for this node is implicitly stored in the combination of the byte
        // width and the length.
        let bytes_per_index = header.bytes_per_index();
        let len = (end - header.start()) / bytes_per_index;

        // LLVM doesn't hoist EdgeHeader::mask so we do it ourselves.
        let mask = header.mask();
        (0..len).map(move |_| {
            // Doing this slicing in this order ensures that the first bounds check suffices for
            // all the others.
            let index = &raw[..DEP_NODE_SIZE];
            raw = &raw[bytes_per_index..];
            let index = u32::from_le_bytes(index.try_into().unwrap()) & mask;
            SerializedDepNodeIndex::from_u32(index)
        })
    }

    #[inline]
    pub fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> DepNode {
        self.nodes[dep_node_index]
    }

    #[inline]
    pub fn node_to_index_opt(&self, dep_node: &DepNode) -> Option<SerializedDepNodeIndex> {
        self.index.get(dep_node.kind.as_usize())?.get(&dep_node.hash).cloned()
    }

    #[inline]
    pub fn fingerprint_by_index(&self, dep_node_index: SerializedDepNodeIndex) -> Fingerprint {
        self.fingerprints[dep_node_index]
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// A packed representation of an edge's start index and byte width.
///
/// This is packed by stealing 2 bits from the start index, which means we only accommodate edge
/// data arrays up to a quarter of our address space. Which seems fine.
#[derive(Debug, Clone, Copy)]
struct EdgeHeader {
    repr: usize,
}

impl EdgeHeader {
    #[inline]
    fn start(self) -> usize {
        self.repr >> DEP_NODE_WIDTH_BITS
    }

    #[inline]
    fn bytes_per_index(self) -> usize {
        (self.repr & mask(DEP_NODE_WIDTH_BITS)) + 1
    }

    #[inline]
    fn mask(self) -> u32 {
        mask(self.bytes_per_index() * 8) as u32
    }
}

#[inline]
fn mask(bits: usize) -> usize {
    usize::MAX >> ((size_of::<usize>() * 8) - bits)
}

impl SerializedDepGraph {
    #[instrument(level = "debug", skip(d, deps))]
    pub fn decode<D: Deps>(d: &mut MemDecoder<'_>, deps: &D) -> Arc<SerializedDepGraph> {
        // The last 16 bytes are the node count and edge count.
        debug!("position: {:?}", d.position());
        let (node_count, edge_count) =
            d.with_position(d.len() - 2 * IntEncodedWithFixedSize::ENCODED_SIZE, |d| {
                debug!("position: {:?}", d.position());
                let node_count = IntEncodedWithFixedSize::decode(d).0 as usize;
                let edge_count = IntEncodedWithFixedSize::decode(d).0 as usize;
                (node_count, edge_count)
            });
        debug!("position: {:?}", d.position());

        debug!(?node_count, ?edge_count);

        let graph_bytes = d.len() - (2 * IntEncodedWithFixedSize::ENCODED_SIZE) - d.position();

        let mut nodes = IndexVec::with_capacity(node_count);
        let mut fingerprints = IndexVec::with_capacity(node_count);
        let mut edge_list_indices = IndexVec::with_capacity(node_count);
        // This estimation assumes that all of the encoded bytes are for the edge lists or for the
        // fixed-size node headers. But that's not necessarily true; if any edge list has a length
        // that spills out of the size we can bit-pack into SerializedNodeHeader then some of the
        // total serialized size is also used by leb128-encoded edge list lengths. Neglecting that
        // contribution to graph_bytes means our estimation of the bytes needed for edge_list_data
        // slightly overshoots. But it cannot overshoot by much; consider that the worse case is
        // for a node with length 64, which means the spilled 1-byte leb128 length is 1 byte of at
        // least (34 byte header + 1 byte len + 64 bytes edge data), which is ~1%. A 2-byte leb128
        // length is about the same fractional overhead and it amortizes for yet greater lengths.
        let mut edge_list_data =
            Vec::with_capacity(graph_bytes - node_count * size_of::<SerializedNodeHeader<D>>());

        for _index in 0..node_count {
            // Decode the header for this edge; the header packs together as many of the fixed-size
            // fields as possible to limit the number of times we update decoder state.
            let node_header =
                SerializedNodeHeader::<D> { bytes: d.read_array(), _marker: PhantomData };

            let _i: SerializedDepNodeIndex = nodes.push(node_header.node());
            debug_assert_eq!(_i.index(), _index);

            let _i: SerializedDepNodeIndex = fingerprints.push(node_header.fingerprint());
            debug_assert_eq!(_i.index(), _index);

            // If the length of this node's edge list is small, the length is stored in the header.
            // If it is not, we fall back to another decoder call.
            let num_edges = node_header.len().unwrap_or_else(|| d.read_u32());

            // The edges index list uses the same varint strategy as rmeta tables; we select the
            // number of byte elements per-array not per-element. This lets us read the whole edge
            // list for a node with one decoder call and also use the on-disk format in memory.
            let edges_len_bytes = node_header.bytes_per_index() * (num_edges as usize);
            // The in-memory structure for the edges list stores the byte width of the edges on
            // this node with the offset into the global edge data array.
            let edges_header = node_header.edges_header(&edge_list_data);

            edge_list_data.extend(d.read_raw_bytes(edges_len_bytes));

            let _i: SerializedDepNodeIndex = edge_list_indices.push(edges_header);
            debug_assert_eq!(_i.index(), _index);
        }

        // When we access the edge list data, we do a fixed-size read from the edge list data then
        // mask off the bytes that aren't for that edge index, so the last read may dangle off the
        // end of the array. This padding ensure it doesn't.
        edge_list_data.extend(&[0u8; DEP_NODE_PAD]);

        // Read the number of each dep kind and use it to create an hash map with a suitable size.
        let mut index: Vec<_> = (0..(D::DEP_KIND_MAX + 1))
            .map(|_| UnhashMap::with_capacity_and_hasher(d.read_u32() as usize, Default::default()))
            .collect();

        for (idx, node) in nodes.iter_enumerated() {
            if index[node.kind.as_usize()].insert(node.hash, idx).is_some() {
                // Side effect nodes can have duplicates
                if node.kind != D::DEP_KIND_SIDE_EFFECT {
                    let name = deps.name(node.kind);
                    panic!(
                    "Error: A dep graph node ({name}) does not have an unique index. \
                     Running a clean build on a nightly compiler with `-Z incremental-verify-ich` \
                     can help narrow down the issue for reporting. A clean build may also work around the issue.\n
                     DepNode: {node:?}"
                )
                }
            }
        }

        Arc::new(SerializedDepGraph {
            nodes,
            fingerprints,
            edge_list_indices,
            edge_list_data,
            index,
        })
    }
}

/// A packed representation of all the fixed-size fields in a `NodeInfo`.
///
/// This stores in one byte array:
/// * The `Fingerprint` in the `NodeInfo`
/// * The `Fingerprint` in `DepNode` that is in this `NodeInfo`
/// * The `DepKind`'s discriminant (a u16, but not all bits are used...)
/// * The byte width of the encoded edges for this node
/// * In whatever bits remain, the length of the edge list for this node, if it fits
struct SerializedNodeHeader<D> {
    // 2 bytes for the DepNode
    // 16 for Fingerprint in DepNode
    // 16 for Fingerprint in NodeInfo
    bytes: [u8; 34],
    _marker: PhantomData<D>,
}

// The fields of a `SerializedNodeHeader`, this struct is an implementation detail and exists only
// to make the implementation of `SerializedNodeHeader` simpler.
struct Unpacked {
    len: Option<u32>,
    bytes_per_index: usize,
    kind: DepKind,
    hash: PackedFingerprint,
    fingerprint: Fingerprint,
}

// Bit fields, where
// M: bits used to store the length of a node's edge list
// N: bits used to store the byte width of elements of the edge list
// are
// 0..M    length of the edge
// M..M+N  bytes per index
// M+N..16 kind
impl<D: Deps> SerializedNodeHeader<D> {
    const TOTAL_BITS: usize = size_of::<DepKind>() * 8;
    const LEN_BITS: usize = Self::TOTAL_BITS - Self::KIND_BITS - Self::WIDTH_BITS;
    const WIDTH_BITS: usize = DEP_NODE_WIDTH_BITS;
    const KIND_BITS: usize = Self::TOTAL_BITS - D::DEP_KIND_MAX.leading_zeros() as usize;
    const MAX_INLINE_LEN: usize = (u16::MAX as usize >> (Self::TOTAL_BITS - Self::LEN_BITS)) - 1;

    #[inline]
    fn new(
        node: DepNode,
        fingerprint: Fingerprint,
        edge_max_index: u32,
        edge_count: usize,
    ) -> Self {
        debug_assert_eq!(Self::TOTAL_BITS, Self::LEN_BITS + Self::WIDTH_BITS + Self::KIND_BITS);

        let mut head = node.kind.as_inner();

        let free_bytes = edge_max_index.leading_zeros() as usize / 8;
        let bytes_per_index = (DEP_NODE_SIZE - free_bytes).saturating_sub(1);
        head |= (bytes_per_index as u16) << Self::KIND_BITS;

        // Encode number of edges + 1 so that we can reserve 0 to indicate that the len doesn't fit
        // in this bitfield.
        if edge_count <= Self::MAX_INLINE_LEN {
            head |= (edge_count as u16 + 1) << (Self::KIND_BITS + Self::WIDTH_BITS);
        }

        let hash: Fingerprint = node.hash.into();

        // Using half-open ranges ensures an unconditional panic if we get the magic numbers wrong.
        let mut bytes = [0u8; 34];
        bytes[..2].copy_from_slice(&head.to_le_bytes());
        bytes[2..18].copy_from_slice(&hash.to_le_bytes());
        bytes[18..].copy_from_slice(&fingerprint.to_le_bytes());

        #[cfg(debug_assertions)]
        {
            let res = Self { bytes, _marker: PhantomData };
            assert_eq!(fingerprint, res.fingerprint());
            assert_eq!(node, res.node());
            if let Some(len) = res.len() {
                assert_eq!(edge_count, len as usize);
            }
        }
        Self { bytes, _marker: PhantomData }
    }

    #[inline]
    fn unpack(&self) -> Unpacked {
        let head = u16::from_le_bytes(self.bytes[..2].try_into().unwrap());
        let hash = self.bytes[2..18].try_into().unwrap();
        let fingerprint = self.bytes[18..].try_into().unwrap();

        let kind = head & mask(Self::KIND_BITS) as u16;
        let bytes_per_index = (head >> Self::KIND_BITS) & mask(Self::WIDTH_BITS) as u16;
        let len = (head as u32) >> (Self::WIDTH_BITS + Self::KIND_BITS);

        Unpacked {
            len: len.checked_sub(1),
            bytes_per_index: bytes_per_index as usize + 1,
            kind: DepKind::new(kind),
            hash: Fingerprint::from_le_bytes(hash).into(),
            fingerprint: Fingerprint::from_le_bytes(fingerprint),
        }
    }

    #[inline]
    fn len(&self) -> Option<u32> {
        self.unpack().len
    }

    #[inline]
    fn bytes_per_index(&self) -> usize {
        self.unpack().bytes_per_index
    }

    #[inline]
    fn fingerprint(&self) -> Fingerprint {
        self.unpack().fingerprint
    }

    #[inline]
    fn node(&self) -> DepNode {
        let Unpacked { kind, hash, .. } = self.unpack();
        DepNode { kind, hash }
    }

    #[inline]
    fn edges_header(&self, edge_list_data: &[u8]) -> EdgeHeader {
        EdgeHeader {
            repr: (edge_list_data.len() << DEP_NODE_WIDTH_BITS) | (self.bytes_per_index() - 1),
        }
    }
}

#[derive(Debug)]
struct NodeInfo {
    node: DepNode,
    fingerprint: Fingerprint,
    edges: EdgesVec,
}

impl NodeInfo {
    fn encode<D: Deps>(&self, e: &mut FileEncoder) {
        let NodeInfo { node, fingerprint, ref edges } = *self;
        let header =
            SerializedNodeHeader::<D>::new(node, fingerprint, edges.max_index(), edges.len());
        e.write_array(header.bytes);

        if header.len().is_none() {
            // The edges are all unique and the number of unique indices is less than u32::MAX.
            e.emit_u32(edges.len().try_into().unwrap());
        }

        let bytes_per_index = header.bytes_per_index();
        for node_index in edges.iter() {
            e.write_with(|dest| {
                *dest = node_index.as_u32().to_le_bytes();
                bytes_per_index
            });
        }
    }

    /// Encode a node that was promoted from the previous graph. It reads the edges directly from
    /// the previous dep graph and expects all edges to already have a new dep node index assigned.
    /// This avoids the overhead of constructing `EdgesVec`, which would be needed to call `encode`.
    #[inline]
    fn encode_promoted<D: Deps>(
        e: &mut FileEncoder,
        node: DepNode,
        fingerprint: Fingerprint,
        prev_index: SerializedDepNodeIndex,
        colors: &DepNodeColorMap,
        previous: &SerializedDepGraph,
    ) -> usize {
        let edges = previous.edge_targets_from(prev_index);
        let edge_count = edges.size_hint().0;

        // Find the highest edge in the new dep node indices
        let edge_max =
            edges.clone().map(|i| colors.current(i).unwrap().as_u32()).max().unwrap_or(0);

        let header = SerializedNodeHeader::<D>::new(node, fingerprint, edge_max, edge_count);
        e.write_array(header.bytes);

        if header.len().is_none() {
            // The edges are all unique and the number of unique indices is less than u32::MAX.
            e.emit_u32(edge_count.try_into().unwrap());
        }

        let bytes_per_index = header.bytes_per_index();
        for node_index in edges {
            let node_index = colors.current(node_index).unwrap();
            e.write_with(|dest| {
                *dest = node_index.as_u32().to_le_bytes();
                bytes_per_index
            });
        }

        edge_count
    }
}

struct Stat {
    kind: DepKind,
    node_counter: u64,
    edge_counter: u64,
}

struct EncoderState<D: Deps> {
    previous: Arc<SerializedDepGraph>,
    encoder: FileEncoder,
    total_node_count: usize,
    total_edge_count: usize,
    stats: Option<FxHashMap<DepKind, Stat>>,

    /// Stores the number of times we've encoded each dep kind.
    kind_stats: Vec<u32>,
    marker: PhantomData<D>,
}

impl<D: Deps> EncoderState<D> {
    fn new(encoder: FileEncoder, record_stats: bool, previous: Arc<SerializedDepGraph>) -> Self {
        Self {
            previous,
            encoder,
            total_edge_count: 0,
            total_node_count: 0,
            stats: record_stats.then(FxHashMap::default),
            kind_stats: iter::repeat(0).take(D::DEP_KIND_MAX as usize + 1).collect(),
            marker: PhantomData,
        }
    }

    #[inline]
    fn record(
        &mut self,
        node: DepNode,
        edge_count: usize,
        edges: impl FnOnce(&mut Self) -> Vec<DepNodeIndex>,
        record_graph: &Option<Lock<DepGraphQuery>>,
    ) -> DepNodeIndex {
        let index = DepNodeIndex::new(self.total_node_count);

        self.total_node_count += 1;
        self.kind_stats[node.kind.as_usize()] += 1;
        self.total_edge_count += edge_count;

        if let Some(record_graph) = &record_graph {
            // Call `edges` before the outlined code to allow the closure to be optimized out.
            let edges = edges(self);

            // Outline the build of the full dep graph as it's typically disabled and cold.
            outline(move || {
                // Do not ICE when a query is called from within `with_query`.
                if let Some(record_graph) = &mut record_graph.try_lock() {
                    record_graph.push(index, node, &edges);
                }
            });
        }

        if let Some(stats) = &mut self.stats {
            let kind = node.kind;

            // Outline the stats code as it's typically disabled and cold.
            outline(move || {
                let stat =
                    stats.entry(kind).or_insert(Stat { kind, node_counter: 0, edge_counter: 0 });
                stat.node_counter += 1;
                stat.edge_counter += edge_count as u64;
            });
        }

        index
    }

    /// Encodes a node to the current graph.
    fn encode_node(
        &mut self,
        node: &NodeInfo,
        record_graph: &Option<Lock<DepGraphQuery>>,
    ) -> DepNodeIndex {
        node.encode::<D>(&mut self.encoder);
        self.record(node.node, node.edges.len(), |_| node.edges[..].to_vec(), record_graph)
    }

    /// Encodes a node that was promoted from the previous graph. It reads the information directly from
    /// the previous dep graph for performance reasons.
    ///
    /// This differs from `encode_node` where you have to explicitly provide the relevant `NodeInfo`.
    ///
    /// It expects all edges to already have a new dep node index assigned.
    #[inline]
    fn encode_promoted_node(
        &mut self,
        prev_index: SerializedDepNodeIndex,
        record_graph: &Option<Lock<DepGraphQuery>>,
        colors: &DepNodeColorMap,
    ) -> DepNodeIndex {
        let node = self.previous.index_to_node(prev_index);

        let fingerprint = self.previous.fingerprint_by_index(prev_index);
        let edge_count = NodeInfo::encode_promoted::<D>(
            &mut self.encoder,
            node,
            fingerprint,
            prev_index,
            colors,
            &self.previous,
        );

        self.record(
            node,
            edge_count,
            |this| {
                this.previous
                    .edge_targets_from(prev_index)
                    .map(|i| colors.current(i).unwrap())
                    .collect()
            },
            record_graph,
        )
    }

    fn finish(self, profiler: &SelfProfilerRef) -> FileEncodeResult {
        let Self {
            mut encoder,
            total_node_count,
            total_edge_count,
            stats: _,
            kind_stats,
            marker: _,
            previous: _,
        } = self;

        let node_count = total_node_count.try_into().unwrap();
        let edge_count = total_edge_count.try_into().unwrap();

        // Encode the number of each dep kind encountered
        for count in kind_stats.iter() {
            count.encode(&mut encoder);
        }

        debug!(?node_count, ?edge_count);
        debug!("position: {:?}", encoder.position());
        IntEncodedWithFixedSize(node_count).encode(&mut encoder);
        IntEncodedWithFixedSize(edge_count).encode(&mut encoder);
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

pub(crate) struct GraphEncoder<D: Deps> {
    profiler: SelfProfilerRef,
    status: Lock<Option<EncoderState<D>>>,
    record_graph: Option<Lock<DepGraphQuery>>,
}

impl<D: Deps> GraphEncoder<D> {
    pub(crate) fn new(
        encoder: FileEncoder,
        prev_node_count: usize,
        record_graph: bool,
        record_stats: bool,
        profiler: &SelfProfilerRef,
        previous: Arc<SerializedDepGraph>,
    ) -> Self {
        let record_graph = record_graph.then(|| Lock::new(DepGraphQuery::new(prev_node_count)));
        let status = Lock::new(Some(EncoderState::new(encoder, record_stats, previous)));
        GraphEncoder { status, record_graph, profiler: profiler.clone() }
    }

    pub(crate) fn with_query(&self, f: impl Fn(&DepGraphQuery)) {
        if let Some(record_graph) = &self.record_graph {
            f(&record_graph.lock())
        }
    }

    pub(crate) fn print_incremental_info(
        &self,
        total_read_count: u64,
        total_duplicate_read_count: u64,
    ) {
        let mut status = self.status.lock();
        let status = status.as_mut().unwrap();
        if let Some(record_stats) = &status.stats {
            let mut stats: Vec<_> = record_stats.values().collect();
            stats.sort_by_key(|s| -(s.node_counter as i64));

            const SEPARATOR: &str = "[incremental] --------------------------------\
                                     ----------------------------------------------\
                                     ------------";

            eprintln!("[incremental]");
            eprintln!("[incremental] DepGraph Statistics");
            eprintln!("{SEPARATOR}");
            eprintln!("[incremental]");
            eprintln!("[incremental] Total Node Count: {}", status.total_node_count);
            eprintln!("[incremental] Total Edge Count: {}", status.total_edge_count);

            if cfg!(debug_assertions) {
                eprintln!("[incremental] Total Edge Reads: {total_read_count}");
                eprintln!("[incremental] Total Duplicate Edge Reads: {total_duplicate_read_count}");
            }

            eprintln!("[incremental]");
            eprintln!(
                "[incremental]  {:<36}| {:<17}| {:<12}| {:<17}|",
                "Node Kind", "Node Frequency", "Node Count", "Avg. Edge Count"
            );
            eprintln!("{SEPARATOR}");

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

            eprintln!("{SEPARATOR}");
            eprintln!("[incremental]");
        }
    }

    /// Encodes a node that does not exists in the previous graph.
    pub(crate) fn send_new(
        &self,
        node: DepNode,
        fingerprint: Fingerprint,
        edges: EdgesVec,
    ) -> DepNodeIndex {
        let _prof_timer = self.profiler.generic_activity("incr_comp_encode_dep_graph");
        let node = NodeInfo { node, fingerprint, edges };
        self.status.lock().as_mut().unwrap().encode_node(&node, &self.record_graph)
    }

    /// Encodes a node that exists in the previous graph, but was re-executed.
    ///
    /// This will also ensure the dep node is colored either red or green.
    pub(crate) fn send_and_color(
        &self,
        prev_index: SerializedDepNodeIndex,
        colors: &DepNodeColorMap,
        node: DepNode,
        fingerprint: Fingerprint,
        edges: EdgesVec,
        is_green: bool,
    ) -> DepNodeIndex {
        let _prof_timer = self.profiler.generic_activity("incr_comp_encode_dep_graph");
        let node = NodeInfo { node, fingerprint, edges };

        let mut status = self.status.lock();
        let status = status.as_mut().unwrap();

        // Check colors inside the lock to avoid racing when `send_promoted` is called concurrently
        // on the same index.
        match colors.get(prev_index) {
            None => {
                let dep_node_index = status.encode_node(&node, &self.record_graph);
                colors.insert(
                    prev_index,
                    if is_green { DepNodeColor::Green(dep_node_index) } else { DepNodeColor::Red },
                );
                dep_node_index
            }
            Some(DepNodeColor::Green(dep_node_index)) => dep_node_index,
            Some(DepNodeColor::Red) => panic!(),
        }
    }

    /// Encodes a node that was promoted from the previous graph. It reads the information directly from
    /// the previous dep graph and expects all edges to already have a new dep node index assigned.
    ///
    /// This will also ensure the dep node is marked green.
    #[inline]
    pub(crate) fn send_promoted(
        &self,
        prev_index: SerializedDepNodeIndex,
        colors: &DepNodeColorMap,
    ) -> DepNodeIndex {
        let _prof_timer = self.profiler.generic_activity("incr_comp_encode_dep_graph");

        let mut status = self.status.lock();
        let status = status.as_mut().unwrap();

        // Check colors inside the lock to avoid racing when `send_promoted` or `send_and_color`
        // is called concurrently on the same index.
        match colors.get(prev_index) {
            None => {
                let dep_node_index =
                    status.encode_promoted_node(prev_index, &self.record_graph, colors);
                colors.insert(prev_index, DepNodeColor::Green(dep_node_index));
                dep_node_index
            }
            Some(DepNodeColor::Green(dep_node_index)) => dep_node_index,
            Some(DepNodeColor::Red) => panic!(),
        }
    }

    pub(crate) fn finish(&self) -> FileEncodeResult {
        let _prof_timer = self.profiler.generic_activity("incr_comp_encode_dep_graph_finish");

        self.status.lock().take().unwrap().finish(&self.profiler)
    }
}
