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
//! The encoding of the dep-graph is generally designed around the fact that fixed-size
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
//!
//! Dep-graph indices are bulk allocated to threads inside `LocalEncoderState`. Having threads
//! own these indices helps avoid races when they are conditionally used when marking nodes green.
//! It also reduces congestion on the shared index count.

use std::cell::RefCell;
use std::cmp::max;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::{iter, mem, u64};

use rustc_data_structures::fingerprint::{Fingerprint, PackedFingerprint};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::outline;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sync::{AtomicU64, Lock, WorkerLocal, broadcast};
use rustc_data_structures::unhash::UnhashMap;
use rustc_index::IndexVec;
use rustc_serialize::opaque::mem_encoder::MemEncoder;
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder, IntEncodedWithFixedSize, MemDecoder};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_session::Session;
use tracing::{debug, instrument};

use super::graph::{CurrentDepGraph, DepNodeColor, DepNodeColorMap};
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
///
/// There may be unused indices with DEP_KIND_NULL in this graph due to batch allocation of
/// indices to threads.
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
    /// The number of previous compilation sessions. This is used to generate
    /// unique anon dep nodes per session.
    session_count: u64,
}

impl SerializedDepGraph {
    #[inline]
    pub fn edge_targets_from(
        &self,
        source: SerializedDepNodeIndex,
    ) -> impl Iterator<Item = SerializedDepNodeIndex> + Clone {
        let header = self.edge_list_indices[source];
        let mut raw = &self.edge_list_data[header.start()..];

        let bytes_per_index = header.bytes_per_index();

        // LLVM doesn't hoist EdgeHeader::mask so we do it ourselves.
        let mask = header.mask();
        (0..header.num_edges).map(move |_| {
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

    #[inline]
    pub fn session_count(&self) -> u64 {
        self.session_count
    }
}

/// A packed representation of an edge's start index and byte width.
///
/// This is packed by stealing 2 bits from the start index, which means we only accommodate edge
/// data arrays up to a quarter of our address space. Which seems fine.
#[derive(Debug, Clone, Copy)]
struct EdgeHeader {
    repr: usize,
    num_edges: u32,
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

        // `node_max` is the number of indices including empty nodes while `node_count`
        // is the number of actually encoded nodes.
        let (node_max, node_count, edge_count) =
            d.with_position(d.len() - 3 * IntEncodedWithFixedSize::ENCODED_SIZE, |d| {
                debug!("position: {:?}", d.position());
                let node_max = IntEncodedWithFixedSize::decode(d).0 as usize;
                let node_count = IntEncodedWithFixedSize::decode(d).0 as usize;
                let edge_count = IntEncodedWithFixedSize::decode(d).0 as usize;
                (node_max, node_count, edge_count)
            });
        debug!("position: {:?}", d.position());

        debug!(?node_count, ?edge_count);

        let graph_bytes = d.len() - (3 * IntEncodedWithFixedSize::ENCODED_SIZE) - d.position();

        let mut nodes = IndexVec::from_elem_n(
            DepNode { kind: D::DEP_KIND_NULL, hash: PackedFingerprint::from(Fingerprint::ZERO) },
            node_max,
        );
        let mut fingerprints = IndexVec::from_elem_n(Fingerprint::ZERO, node_max);
        let mut edge_list_indices =
            IndexVec::from_elem_n(EdgeHeader { repr: 0, num_edges: 0 }, node_max);

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

        for _ in 0..node_count {
            // Decode the header for this edge; the header packs together as many of the fixed-size
            // fields as possible to limit the number of times we update decoder state.
            let node_header =
                SerializedNodeHeader::<D> { bytes: d.read_array(), _marker: PhantomData };

            let index = node_header.index();

            let node = &mut nodes[index];
            // Make sure there's no duplicate indices in the dep graph.
            assert!(node_header.node().kind != D::DEP_KIND_NULL && node.kind == D::DEP_KIND_NULL);
            *node = node_header.node();

            fingerprints[index] = node_header.fingerprint();

            // If the length of this node's edge list is small, the length is stored in the header.
            // If it is not, we fall back to another decoder call.
            let num_edges = node_header.len().unwrap_or_else(|| d.read_u32());

            // The edges index list uses the same varint strategy as rmeta tables; we select the
            // number of byte elements per-array not per-element. This lets us read the whole edge
            // list for a node with one decoder call and also use the on-disk format in memory.
            let edges_len_bytes = node_header.bytes_per_index() * (num_edges as usize);
            // The in-memory structure for the edges list stores the byte width of the edges on
            // this node with the offset into the global edge data array.
            let edges_header = node_header.edges_header(&edge_list_data, num_edges);

            edge_list_data.extend(d.read_raw_bytes(edges_len_bytes));

            edge_list_indices[index] = edges_header;
        }

        // When we access the edge list data, we do a fixed-size read from the edge list data then
        // mask off the bytes that aren't for that edge index, so the last read may dangle off the
        // end of the array. This padding ensure it doesn't.
        edge_list_data.extend(&[0u8; DEP_NODE_PAD]);

        // Read the number of each dep kind and use it to create an hash map with a suitable size.
        let mut index: Vec<_> = (0..(D::DEP_KIND_MAX + 1))
            .map(|_| UnhashMap::with_capacity_and_hasher(d.read_u32() as usize, Default::default()))
            .collect();

        let session_count = d.read_u64();

        for (idx, node) in nodes.iter_enumerated() {
            if index[node.kind.as_usize()].insert(node.hash, idx).is_some() {
                // Empty nodes and side effect nodes can have duplicates
                if node.kind != D::DEP_KIND_NULL && node.kind != D::DEP_KIND_SIDE_EFFECT {
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
            session_count,
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
    // 4 bytes for the index
    // 16 for Fingerprint in DepNode
    // 16 for Fingerprint in NodeInfo
    bytes: [u8; 38],
    _marker: PhantomData<D>,
}

// The fields of a `SerializedNodeHeader`, this struct is an implementation detail and exists only
// to make the implementation of `SerializedNodeHeader` simpler.
struct Unpacked {
    len: Option<u32>,
    bytes_per_index: usize,
    kind: DepKind,
    index: SerializedDepNodeIndex,
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
        index: DepNodeIndex,
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
        let mut bytes = [0u8; 38];
        bytes[..2].copy_from_slice(&head.to_le_bytes());
        bytes[2..6].copy_from_slice(&index.as_u32().to_le_bytes());
        bytes[6..22].copy_from_slice(&hash.to_le_bytes());
        bytes[22..].copy_from_slice(&fingerprint.to_le_bytes());

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
        let index = u32::from_le_bytes(self.bytes[2..6].try_into().unwrap());
        let hash = self.bytes[6..22].try_into().unwrap();
        let fingerprint = self.bytes[22..].try_into().unwrap();

        let kind = head & mask(Self::KIND_BITS) as u16;
        let bytes_per_index = (head >> Self::KIND_BITS) & mask(Self::WIDTH_BITS) as u16;
        let len = (head as u32) >> (Self::WIDTH_BITS + Self::KIND_BITS);

        Unpacked {
            len: len.checked_sub(1),
            bytes_per_index: bytes_per_index as usize + 1,
            kind: DepKind::new(kind),
            index: SerializedDepNodeIndex::from_u32(index),
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
    fn index(&self) -> SerializedDepNodeIndex {
        self.unpack().index
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
    fn edges_header(&self, edge_list_data: &[u8], num_edges: u32) -> EdgeHeader {
        EdgeHeader {
            repr: (edge_list_data.len() << DEP_NODE_WIDTH_BITS) | (self.bytes_per_index() - 1),
            num_edges,
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
    fn encode<D: Deps>(&self, e: &mut MemEncoder, index: DepNodeIndex) {
        let NodeInfo { node, fingerprint, ref edges } = *self;
        let header = SerializedNodeHeader::<D>::new(
            node,
            index,
            fingerprint,
            edges.max_index(),
            edges.len(),
        );
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
        e: &mut MemEncoder,
        node: DepNode,
        index: DepNodeIndex,
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

        let header = SerializedNodeHeader::<D>::new(node, index, fingerprint, edge_max, edge_count);
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

struct LocalEncoderState {
    next_node_index: u32,
    remaining_node_index: u32,
    encoder: MemEncoder,
    node_count: usize,
    edge_count: usize,

    /// Stores the number of times we've encoded each dep kind.
    kind_stats: Vec<u32>,
}

struct LocalEncoderResult {
    node_max: u32,
    node_count: usize,
    edge_count: usize,

    /// Stores the number of times we've encoded each dep kind.
    kind_stats: Vec<u32>,
}

struct EncoderState<D: Deps> {
    next_node_index: AtomicU64,
    previous: Arc<SerializedDepGraph>,
    file: Lock<Option<FileEncoder>>,
    local: WorkerLocal<RefCell<LocalEncoderState>>,
    stats: Option<Lock<FxHashMap<DepKind, Stat>>>,
    marker: PhantomData<D>,
}

impl<D: Deps> EncoderState<D> {
    fn new(encoder: FileEncoder, record_stats: bool, previous: Arc<SerializedDepGraph>) -> Self {
        Self {
            previous,
            next_node_index: AtomicU64::new(0),
            stats: record_stats.then(|| Lock::new(FxHashMap::default())),
            file: Lock::new(Some(encoder)),
            local: WorkerLocal::new(|_| {
                RefCell::new(LocalEncoderState {
                    next_node_index: 0,
                    remaining_node_index: 0,
                    edge_count: 0,
                    node_count: 0,
                    encoder: MemEncoder::new(),
                    kind_stats: iter::repeat(0).take(D::DEP_KIND_MAX as usize + 1).collect(),
                })
            }),
            marker: PhantomData,
        }
    }

    #[inline]
    fn next_index(&self, local: &mut LocalEncoderState) -> DepNodeIndex {
        if local.remaining_node_index == 0 {
            const COUNT: u32 = 256;

            // We assume that there won't be enough active threads to overflow `u64` from `u32::MAX` here.
            // This can exceed u32::MAX by at most `N` * `COUNT` where `N` is the thread pool count since
            // `try_into().unwrap()` will make threads panic when `self.next_node_index` exceeds u32::MAX.
            local.next_node_index =
                self.next_node_index.fetch_add(COUNT as u64, Ordering::Relaxed).try_into().unwrap();

            // Check that we'll stay within `u32`
            local.next_node_index.checked_add(COUNT).unwrap();

            local.remaining_node_index = COUNT;
        }

        DepNodeIndex::from_u32(local.next_node_index)
    }

    /// Marks the index previously returned by `next_index` as used.
    #[inline]
    fn bump_index(&self, local: &mut LocalEncoderState) {
        local.remaining_node_index -= 1;
        local.next_node_index += 1;
        local.node_count += 1;
    }

    #[inline]
    fn record(
        &self,
        node: DepNode,
        index: DepNodeIndex,
        edge_count: usize,
        edges: impl FnOnce(&Self) -> Vec<DepNodeIndex>,
        record_graph: &Option<Lock<DepGraphQuery>>,
        local: &mut LocalEncoderState,
    ) {
        local.kind_stats[node.kind.as_usize()] += 1;
        local.edge_count += edge_count;

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

        if let Some(stats) = &self.stats {
            let kind = node.kind;

            // Outline the stats code as it's typically disabled and cold.
            outline(move || {
                let mut stats = stats.lock();
                let stat =
                    stats.entry(kind).or_insert(Stat { kind, node_counter: 0, edge_counter: 0 });
                stat.node_counter += 1;
                stat.edge_counter += edge_count as u64;
            });
        }
    }

    #[inline]
    fn flush_mem_encoder(&self, local: &mut LocalEncoderState) {
        let data = &mut local.encoder.data;
        if data.len() > 64 * 1024 {
            self.file.lock().as_mut().unwrap().emit_raw_bytes(&data[..]);
            data.clear();
        }
    }

    /// Encodes a node to the current graph.
    fn encode_node(
        &self,
        index: DepNodeIndex,
        node: &NodeInfo,
        record_graph: &Option<Lock<DepGraphQuery>>,
        local: &mut LocalEncoderState,
    ) {
        node.encode::<D>(&mut local.encoder, index);
        self.flush_mem_encoder(&mut *local);
        self.record(
            node.node,
            index,
            node.edges.len(),
            |_| node.edges[..].to_vec(),
            record_graph,
            &mut *local,
        );
    }

    /// Encodes a node that was promoted from the previous graph. It reads the information directly from
    /// the previous dep graph for performance reasons.
    ///
    /// This differs from `encode_node` where you have to explicitly provide the relevant `NodeInfo`.
    ///
    /// It expects all edges to already have a new dep node index assigned.
    #[inline]
    fn encode_promoted_node(
        &self,
        index: DepNodeIndex,
        prev_index: SerializedDepNodeIndex,
        record_graph: &Option<Lock<DepGraphQuery>>,
        colors: &DepNodeColorMap,
        local: &mut LocalEncoderState,
    ) {
        let node = self.previous.index_to_node(prev_index);
        let fingerprint = self.previous.fingerprint_by_index(prev_index);
        let edge_count = NodeInfo::encode_promoted::<D>(
            &mut local.encoder,
            node,
            index,
            fingerprint,
            prev_index,
            colors,
            &self.previous,
        );
        self.flush_mem_encoder(&mut *local);
        self.record(
            node,
            index,
            edge_count,
            |this| {
                this.previous
                    .edge_targets_from(prev_index)
                    .map(|i| colors.current(i).unwrap())
                    .collect()
            },
            record_graph,
            &mut *local,
        );
    }

    fn finish(&self, profiler: &SelfProfilerRef, current: &CurrentDepGraph<D>) -> FileEncodeResult {
        // Prevent more indices from being allocated.
        self.next_node_index.store(u32::MAX as u64 + 1, Ordering::SeqCst);

        let results = broadcast(|_| {
            let mut local = self.local.borrow_mut();

            // Prevent more indices from being allocated on this thread.
            local.remaining_node_index = 0;

            let data = mem::replace(&mut local.encoder.data, Vec::new());
            self.file.lock().as_mut().unwrap().emit_raw_bytes(&data);

            LocalEncoderResult {
                kind_stats: local.kind_stats.clone(),
                node_max: local.next_node_index,
                node_count: local.node_count,
                edge_count: local.edge_count,
            }
        });

        let mut encoder = self.file.lock().take().unwrap();

        let mut kind_stats: Vec<u32> = iter::repeat(0).take(D::DEP_KIND_MAX as usize + 1).collect();

        let mut node_max = 0;
        let mut node_count = 0;
        let mut edge_count = 0;

        for result in results {
            node_max = max(node_max, result.node_max);
            node_count += result.node_count;
            edge_count += result.edge_count;
            for (i, stat) in result.kind_stats.iter().enumerate() {
                kind_stats[i] += stat;
            }
        }

        // Encode the number of each dep kind encountered
        for count in kind_stats.iter() {
            count.encode(&mut encoder);
        }

        self.previous.session_count.checked_add(1).unwrap().encode(&mut encoder);

        debug!(?node_max, ?node_count, ?edge_count);
        debug!("position: {:?}", encoder.position());
        IntEncodedWithFixedSize(node_max.try_into().unwrap()).encode(&mut encoder);
        IntEncodedWithFixedSize(node_count.try_into().unwrap()).encode(&mut encoder);
        IntEncodedWithFixedSize(edge_count.try_into().unwrap()).encode(&mut encoder);
        debug!("position: {:?}", encoder.position());
        // Drop the encoder so that nothing is written after the counts.
        let result = encoder.finish();
        if let Ok(position) = result {
            // FIXME(rylev): we hardcode the dep graph file name so we
            // don't need a dependency on rustc_incremental just for that.
            profiler.artifact_size("dep_graph", "dep-graph.bin", position as u64);
        }

        self.print_incremental_info(current, node_count, edge_count);

        result
    }

    fn print_incremental_info(
        &self,
        current: &CurrentDepGraph<D>,
        total_node_count: usize,
        total_edge_count: usize,
    ) {
        if let Some(record_stats) = &self.stats {
            let record_stats = record_stats.lock();
            // `stats` is sorted below so we can allow this lint here.
            #[allow(rustc::potential_query_instability)]
            let mut stats: Vec<_> = record_stats.values().collect();
            stats.sort_by_key(|s| -(s.node_counter as i64));

            const SEPARATOR: &str = "[incremental] --------------------------------\
                                     ----------------------------------------------\
                                     ------------";

            eprintln!("[incremental]");
            eprintln!("[incremental] DepGraph Statistics");
            eprintln!("{SEPARATOR}");
            eprintln!("[incremental]");
            eprintln!("[incremental] Total Node Count: {}", total_node_count);
            eprintln!("[incremental] Total Edge Count: {}", total_edge_count);

            if cfg!(debug_assertions) {
                let total_read_count = current.total_read_count.load(Ordering::Relaxed);
                let total_duplicate_read_count =
                    current.total_duplicate_read_count.load(Ordering::Relaxed);
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

            eprintln!("{SEPARATOR}");
            eprintln!("[incremental]");
        }
    }
}

pub(crate) struct GraphEncoder<D: Deps> {
    profiler: SelfProfilerRef,
    status: EncoderState<D>,
    record_graph: Option<Lock<DepGraphQuery>>,
}

impl<D: Deps> GraphEncoder<D> {
    pub(crate) fn new(
        sess: &Session,
        encoder: FileEncoder,
        prev_node_count: usize,
        previous: Arc<SerializedDepGraph>,
    ) -> Self {
        let record_graph = sess
            .opts
            .unstable_opts
            .query_dep_graph
            .then(|| Lock::new(DepGraphQuery::new(prev_node_count)));
        let status = EncoderState::new(encoder, sess.opts.unstable_opts.incremental_info, previous);
        GraphEncoder { status, record_graph, profiler: sess.prof.clone() }
    }

    pub(crate) fn with_query(&self, f: impl Fn(&DepGraphQuery)) {
        if let Some(record_graph) = &self.record_graph {
            f(&record_graph.lock())
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
        let mut local = self.status.local.borrow_mut();
        let index = self.status.next_index(&mut *local);
        self.status.bump_index(&mut *local);
        self.status.encode_node(index, &node, &self.record_graph, &mut *local);
        index
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

        let mut local = self.status.local.borrow_mut();

        let index = self.status.next_index(&mut *local);

        if is_green {
            // Use `try_mark_green` to avoid racing when `send_promoted` is called concurrently
            // on the same index.
            match colors.try_mark_green(prev_index, index) {
                Ok(()) => (),
                Err(dep_node_index) => return dep_node_index,
            }
        } else {
            colors.insert(prev_index, DepNodeColor::Red);
        }

        self.status.bump_index(&mut *local);
        self.status.encode_node(index, &node, &self.record_graph, &mut *local);
        index
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

        let mut local = self.status.local.borrow_mut();
        let index = self.status.next_index(&mut *local);

        // Use `try_mark_green` to avoid racing when `send_promoted` or `send_and_color`
        // is called concurrently on the same index.
        match colors.try_mark_green(prev_index, index) {
            Ok(()) => {
                self.status.bump_index(&mut *local);
                self.status.encode_promoted_node(
                    index,
                    prev_index,
                    &self.record_graph,
                    colors,
                    &mut *local,
                );
                index
            }
            Err(dep_node_index) => dep_node_index,
        }
    }

    pub(crate) fn finish(&self, current: &CurrentDepGraph<D>) -> FileEncodeResult {
        let _prof_timer = self.profiler.generic_activity("incr_comp_encode_dep_graph_finish");

        self.status.finish(&self.profiler, current)
    }
}
