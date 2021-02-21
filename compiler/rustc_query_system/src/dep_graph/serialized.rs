//! The data that we will serialize and deserialize.

use super::query::DepGraphQuery;
use super::{DepKind, DepNode};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::{Idx, IndexVec};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use std::convert::TryInto;
use std::ops::Range;

#[derive(Debug, PartialEq, Eq)]
pub enum DepNodeColor {
    Green,
    Red,
    New,
}

const TAG_UNKNOWN: u32 = 0;
const TAG_GREEN: u32 = 1 << 30;
const TAG_RED: u32 = 2 << 30;
const TAG_NEW: u32 = 3 << 30;
const TAG_MASK: u32 = TAG_UNKNOWN | TAG_GREEN | TAG_RED | TAG_NEW;
const OFFSET_MASK: u32 = !TAG_MASK;

impl DepNodeColor {
    const fn tag(self) -> u32 {
        match self {
            Self::Green => TAG_GREEN,
            Self::Red => TAG_RED,
            Self::New => TAG_NEW,
        }
    }
}

// The maximum value of `SerializedDepNodeIndex` leaves the upper two bits
// unused so that we can store the node color along with it.
rustc_index::newtype_index! {
    pub struct SerializedDepNodeIndex {
        MAX = 0x7FFF_FFFF
    }
}

// This newtype exists to ensure the main algorithms do not forget interning nodes.
rustc_index::newtype_index! {
    pub struct DepNodeIndex {
        MAX = 0x7FFF_FFFF
    }
}

// Index type for new nodes.
rustc_index::newtype_index! {
    struct SplitIndex {
        MAX = 0x7FFF_FFFF
    }
}

impl SerializedDepNodeIndex {
    pub(super) fn rejuvenate(self) -> DepNodeIndex {
        DepNodeIndex::new(self.index())
    }
}

// We store a large collection of these `edge_list_data`.
// Non-full incremental builds, and want to ensure that the
// element size doesn't inadvertently increase.
static_assert_size!(Option<DepNodeIndex>, 4);
static_assert_size!(Option<SerializedDepNodeIndex>, 4);

#[derive(Copy, Clone, Encodable, Decodable)]
struct ColorAndOffset(u32, u32);

impl std::fmt::Debug for ColorAndOffset {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("ColorAndOffset")
            .field("color", &self.color())
            .field("lifted", &self.lifted())
            .field("range", &self.range())
            .finish()
    }
}

impl ColorAndOffset {
    fn unknown(start: u32, end: u32) -> ColorAndOffset {
        debug_assert_eq!(start & TAG_MASK, 0);
        debug_assert_eq!(end & TAG_MASK, 0);
        ColorAndOffset(start | TAG_UNKNOWN, end | TAG_UNKNOWN)
    }

    #[allow(dead_code)]
    fn new(color: DepNodeColor, range: Range<usize>) -> ColorAndOffset {
        let start: u32 = range.start.try_into().unwrap();
        let end: u32 = range.end.try_into().unwrap();
        debug_assert_eq!(start & TAG_MASK, 0);
        debug_assert_eq!(end & TAG_MASK, 0);
        ColorAndOffset(start | color.tag(), end | TAG_UNKNOWN)
    }

    fn new_lifted(color: DepNodeColor, range: Range<usize>) -> ColorAndOffset {
        let start: u32 = range.start.try_into().unwrap();
        let end: u32 = range.end.try_into().unwrap();
        debug_assert_eq!(start & TAG_MASK, 0);
        debug_assert_eq!(end & TAG_MASK, 0);
        ColorAndOffset(start | color.tag(), end | TAG_GREEN)
    }

    fn set_color(&mut self, color: DepNodeColor) {
        let offset = self.0 & OFFSET_MASK;
        self.0 = color.tag() | offset;
    }

    fn color(self) -> Option<DepNodeColor> {
        let tag = self.0 & TAG_MASK;
        match tag {
            TAG_NEW => Some(DepNodeColor::New),
            TAG_RED => Some(DepNodeColor::Red),
            TAG_GREEN => Some(DepNodeColor::Green),
            TAG_UNKNOWN => None,
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    fn lifted(self) -> bool {
        let tag = self.1 & TAG_MASK;
        match tag {
            TAG_UNKNOWN => false,
            _ => true,
        }
    }

    fn range(self) -> Range<usize> {
        let start = (self.0 & OFFSET_MASK) as usize;
        let end = (self.1 & OFFSET_MASK) as usize;
        start..end
    }
}

fn shrink_range(range: Range<usize>) -> Range<u32> {
    range.start.try_into().unwrap()..range.end.try_into().unwrap()
}

/// Data for use when recompiling the **previous crate**.
///
/// Those IndexVec are never pushed to, so as to avoid large reallocations.
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
    edge_list_indices: IndexVec<SerializedDepNodeIndex, ColorAndOffset>,
    /// A flattened list of all edge targets in the graph. Edge sources are
    /// implicit in edge_list_indices.
    edge_list_data: Vec<DepNodeIndex>,
}

/// Data for use when recompiling the **current crate**.
#[derive(Debug)]
pub struct CurrentDepGraph<K: DepKind> {
    /// The previous graph.
    serialized: SerializedDepGraph<K>,
    /// The set of all DepNodes in the graph
    nodes: IndexVec<SplitIndex, DepNode<K>>,
    /// The set of all Fingerprints in the graph. Each Fingerprint corresponds to
    /// the DepNode at the same index in the nodes vector.
    fingerprints: IndexVec<SplitIndex, Fingerprint>,
    /// For each DepNode, stores the list of edges originating from that
    /// DepNode. Encoded as a [start, end) pair indexing into edge_list_data,
    /// which holds the actual DepNodeIndices of the target nodes.
    edge_list_indices: IndexVec<SplitIndex, Range<u32>>,
    /// A flattened list of all edge targets in the graph. Edge sources are
    /// implicit in edge_list_indices.
    edge_list_data: Vec<DepNodeIndex>,
    /// Reverse map for `nodes`. It is computed on the fly at decoding time.
    index: FxHashMap<DepNode<K>, DepNodeIndex>,
}

impl<K: DepKind> Default for SerializedDepGraph<K> {
    fn default() -> Self {
        Self {
            nodes: IndexVec::new(),
            fingerprints: IndexVec::new(),
            edge_list_indices: IndexVec::new(),
            edge_list_data: Vec::new(),
        }
    }
}

impl<K: DepKind> CurrentDepGraph<K> {
    pub(crate) fn new(serialized: SerializedDepGraph<K>) -> Self {
        let prev_graph_node_count = serialized.nodes.len();
        let nodes = node_count_estimate(prev_graph_node_count);
        let edges = edge_count_estimate(prev_graph_node_count);

        let mut index = FxHashMap::default();
        for (idx, &dep_node) in serialized.nodes.iter_enumerated() {
            debug!("DECODE index={:?} node={:?}", idx, dep_node);
            let _o = index.insert(dep_node, idx.rejuvenate());
            debug_assert_eq!(_o, None);
        }
        Self {
            serialized,
            nodes: IndexVec::with_capacity(nodes),
            fingerprints: IndexVec::with_capacity(nodes),
            edge_list_indices: IndexVec::with_capacity(nodes),
            edge_list_data: Vec::with_capacity(edges),
            index,
        }
    }

    fn intern_new_node(
        &mut self,
        node: DepNode<K>,
        deps: &[DepNodeIndex],
        fingerprint: Fingerprint,
    ) -> DepNodeIndex {
        let index = self.nodes.push(node);
        debug!("intern_new: {:?} {:?}", self.from_split(index), node);
        let _index = self.fingerprints.push(fingerprint);
        debug_assert_eq!(index, _index);
        let range = self.insert_deps(deps);
        let _index = self.edge_list_indices.push(shrink_range(range));
        debug_assert_eq!(index, _index);
        let index = self.from_split(index);
        let _o = self.index.insert(node, index);
        debug_assert_eq!(_o, None);
        index
    }

    fn insert_deps(&mut self, deps: &[DepNodeIndex]) -> Range<usize> {
        let start = self.edge_list_data.len();
        self.edge_list_data.extend(deps.iter().copied());
        let end = self.edge_list_data.len();
        start..end
    }

    fn update_deps(
        &mut self,
        index: SerializedDepNodeIndex,
        color: DepNodeColor,
        deps: &[DepNodeIndex],
    ) {
        debug!("intern_color: {:?} => {:?}", index, color);
        let range = self.serialized.edge_list_indices[index];
        debug_assert_eq!(range.color(), None);
        let range = self.insert_deps(deps);
        let range = ColorAndOffset::new_lifted(color, range);
        self.serialized.edge_list_indices[index] = range;
    }

    pub(crate) fn intern_dark_green_node(&mut self, index: SerializedDepNodeIndex) -> DepNodeIndex {
        debug!("intern_drak_green: {:?}", index);
        debug_assert_eq!(self.serialized.edge_list_indices[index].color(), None);
        self.serialized.edge_list_indices[index].set_color(DepNodeColor::Green);
        debug!("intern_color: {:?} => Green", index);
        index.rejuvenate()
    }

    pub(crate) fn intern_anon_node(
        &mut self,
        node: DepNode<K>,
        deps: &[DepNodeIndex],
    ) -> DepNodeIndex {
        self.dep_node_index_of_opt(&node)
            .unwrap_or_else(|| self.intern_new_node(node, deps, Fingerprint::ZERO))
    }

    pub(crate) fn intern_task_node(
        &mut self,
        node: DepNode<K>,
        deps: &[DepNodeIndex],
        fingerprint: Option<Fingerprint>,
        print_status: bool,
    ) -> DepNodeIndex {
        let print_status = cfg!(debug_assertions) && print_status;

        if let Some(&existing) = self.index.get(&node) {
            let prev_index = self
                .as_serialized(existing)
                .unwrap_or_else(|_| panic!("Node {:?} is being interned multiple times.", node));
            match self.color(prev_index) {
                Some(DepNodeColor::Red) | Some(DepNodeColor::New) => {
                    panic!("Node {:?} is being interned multiple times.", node)
                }

                // This can happen when trying to compute the result of green queries.
                Some(DepNodeColor::Green) => return existing,

                None => {}
            }

            // Determine the color and index of the new `DepNode`.
            let color = if let Some(fingerprint) = fingerprint {
                if fingerprint == self.serialized.fingerprints[prev_index] {
                    if print_status {
                        eprintln!("[task::green] {:?}", node);
                    }

                    // This is a light green node: it existed in the previous compilation,
                    // its query was re-executed, and it has the same result as before.
                    DepNodeColor::Green
                } else {
                    if print_status {
                        eprintln!("[task::red] {:?}", node);
                    }

                    // This is a red node: it existed in the previous compilation, its query
                    // was re-executed, but it has a different result from before.
                    self.serialized.fingerprints[prev_index] = fingerprint;
                    DepNodeColor::Red
                }
            } else {
                if print_status {
                    eprintln!("[task::red] {:?}", node);
                }

                // This is a red node, effectively: it existed in the previous compilation
                // session, its query was re-executed, but it doesn't compute a result hash
                // (i.e. it represents a `no_hash` query), so we have no way of determining
                // whether or not the result was the same as before.
                self.serialized.fingerprints[prev_index] = Fingerprint::ZERO;
                DepNodeColor::Red
            };

            self.update_deps(prev_index, color, deps);
            prev_index.rejuvenate()
        } else {
            if print_status {
                eprintln!("[task::new] {:?}", node);
            }

            // This is a new node: it didn't exist in the previous compilation session.
            self.intern_new_node(node, deps, fingerprint.unwrap_or(Fingerprint::ZERO))
        }
    }

    #[inline]
    fn as_serialized(&self, index: DepNodeIndex) -> Result<SerializedDepNodeIndex, SplitIndex> {
        let index = index.index();
        let count = self.serialized.nodes.len();
        if index < count {
            Ok(SerializedDepNodeIndex::new(index))
        } else {
            Err(SplitIndex::new(index - count))
        }
    }

    #[inline]
    fn from_split(&self, index: SplitIndex) -> DepNodeIndex {
        DepNodeIndex::new(self.serialized.nodes.len() + index.index())
    }

    #[inline]
    fn serialized_edges(&self, source: SerializedDepNodeIndex) -> &[DepNodeIndex] {
        let range = self.serialized.edge_list_indices[source];
        if range.lifted() {
            &self.edge_list_data[range.range()]
        } else {
            &self.serialized.edge_list_data[range.range()]
        }
    }

    #[inline]
    fn new_edges(&self, source: SplitIndex) -> &[DepNodeIndex] {
        let range = &self.edge_list_indices[source];
        let start = range.start as usize;
        let end = range.end as usize;
        &self.edge_list_data[start..end]
    }

    #[inline]
    pub(crate) fn color_or_edges(
        &self,
        source: SerializedDepNodeIndex,
    ) -> Result<DepNodeColor, &'static [SerializedDepNodeIndex]> {
        let range = self.serialized.edge_list_indices[source];
        if let Some(color) = range.color() {
            return Ok(color);
        }
        // The node has not been colored, so the dependencies have not been lifted to point to the
        // new nodes vector.
        debug_assert!(!range.lifted());
        let edges = &self.serialized.edge_list_data[range.range()];
        debug_assert_eq!(
            std::mem::size_of::<DepNodeIndex>(),
            std::mem::size_of::<SerializedDepNodeIndex>()
        );
        // SAFETY: 1. serialized.edge_list_data is never modified.
        // 2. SerializedDepNodeIndex and DepNodeIndex have the same binary representation.
        let edges = unsafe { std::mem::transmute::<&[_], &[_]>(edges) };
        Err(edges)
    }

    #[inline]
    pub(crate) fn edge_targets_from(&self, source: DepNodeIndex) -> &[DepNodeIndex] {
        match self.as_serialized(source) {
            Ok(source) => self.serialized_edges(source),
            Err(source) => self.new_edges(source),
        }
    }

    #[inline]
    pub(crate) fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> DepNode<K> {
        self.serialized.nodes[dep_node_index]
    }

    #[inline]
    pub(crate) fn dep_node_of(&self, dep_node_index: DepNodeIndex) -> DepNode<K> {
        match self.as_serialized(dep_node_index) {
            Ok(serialized) => self.serialized.nodes[serialized],
            Err(new) => self.nodes[new],
        }
    }

    #[inline]
    pub(crate) fn node_to_index_opt(
        &self,
        dep_node: &DepNode<K>,
    ) -> Option<SerializedDepNodeIndex> {
        let idx = *self.index.get(dep_node)?;
        self.as_serialized(idx).ok()
    }

    #[inline]
    pub(crate) fn dep_node_index_of_opt(&self, dep_node: &DepNode<K>) -> Option<DepNodeIndex> {
        let index = *self.index.get(dep_node)?;
        if let Ok(prev) = self.as_serialized(index) {
            // Return none if the node has not been coloured yet.
            self.serialized.edge_list_indices[prev].color()?;
        }
        Some(index)
    }

    #[inline]
    pub(crate) fn color(&self, index: SerializedDepNodeIndex) -> Option<DepNodeColor> {
        self.serialized.edge_list_indices[index].color()
    }

    #[inline]
    pub(crate) fn fingerprint_of(&self, dep_node_index: DepNodeIndex) -> Fingerprint {
        match self.as_serialized(dep_node_index) {
            Ok(serialized) => self.serialized.fingerprints[serialized],
            Err(split) => self.fingerprints[split],
        }
    }

    #[inline]
    pub(crate) fn serialized_indices(&self) -> impl Iterator<Item = SerializedDepNodeIndex> {
        self.serialized.nodes.indices()
    }

    #[inline]
    fn live_serialized_indices(&self) -> impl Iterator<Item = SerializedDepNodeIndex> + '_ {
        self.serialized.edge_list_indices.iter_enumerated().filter_map(|(i, range)| {
            // Return none if the node has not been coloured yet.
            let _ = range.color()?;
            Some(i)
        })
    }

    #[inline]
    fn new_indices(&self) -> impl Iterator<Item = DepNodeIndex> + '_ {
        self.nodes.indices().map(move |i| self.from_split(i))
    }

    #[inline]
    pub(crate) fn live_indices(&self) -> impl Iterator<Item = DepNodeIndex> + '_ {
        // New indices are always live.
        self.live_serialized_indices()
            .map(SerializedDepNodeIndex::rejuvenate)
            .chain(self.new_indices())
    }

    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        self.live_indices().count()
    }

    #[inline]
    fn edge_map(&self) -> impl Iterator<Item = &[DepNodeIndex]> + '_ {
        let serialized_edges =
            self.live_serialized_indices().map(move |index| self.serialized_edges(index));
        let new_edges = self.edge_list_indices.iter().map(move |range| {
            let start = range.start as usize;
            let end = range.end as usize;
            &self.edge_list_data[start..end]
        });
        serialized_edges.chain(new_edges)
    }

    #[inline]
    pub(crate) fn edge_count(&self) -> usize {
        self.edge_map().flatten().count()
    }

    pub(crate) fn query(&self) -> DepGraphQuery<K> {
        let node_count = self.node_count();
        let edge_count = self.edge_count();

        let mut nodes = Vec::with_capacity(node_count);
        nodes.extend(self.live_indices().map(|i| self.dep_node_of(i)));

        let mut edge_list_indices = Vec::with_capacity(node_count);
        let mut edge_list_data = Vec::with_capacity(edge_count);
        for edges in self.edge_map() {
            let start = edge_list_data.len();
            edge_list_data.extend(edges.iter().map(|i| i.index() as usize));
            let end = edge_list_data.len();
            edge_list_indices.push((start, end))
        }

        debug_assert_eq!(nodes.len(), edge_list_indices.len());
        DepGraphQuery::new(&nodes[..], &edge_list_indices[..], &edge_list_data[..])
    }

    pub(crate) fn compression_map(&self) -> IndexVec<DepNodeIndex, Option<SerializedDepNodeIndex>> {
        let mut new_index = SerializedDepNodeIndex::new(0);
        let mut remap = IndexVec::from_elem_n(None, self.serialized.nodes.len() + self.nodes.len());
        for index in self.live_indices() {
            debug_assert!(new_index.index() <= index.index());
            remap[index] = Some(new_index);
            new_index.increment_by(1);
        }
        remap
    }
}

impl<E: Encoder, K: DepKind + Encodable<E>> Encodable<E> for CurrentDepGraph<K> {
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        let remap = self.compression_map();
        let live_indices = || remap.iter_enumerated().filter_map(|(s, &n)| Some((s, n?)));
        let node_count = live_indices().count();
        let edge_count = self.edge_count();

        e.emit_struct("SerializedDepGraph", 4, |e| {
            e.emit_struct_field("nodes", 0, |e| {
                e.emit_seq(node_count, |e| {
                    for (index, new_index) in live_indices() {
                        let node = self.dep_node_of(index);
                        e.emit_seq_elt(new_index.index(), |e| node.encode(e))?;
                    }
                    Ok(())
                })
            })?;
            e.emit_struct_field("fingerprints", 1, |e| {
                e.emit_seq(node_count, |e| {
                    for (index, new_index) in live_indices() {
                        let node = self.fingerprint_of(index);
                        e.emit_seq_elt(new_index.index(), |e| node.encode(e))?;
                    }
                    Ok(())
                })
            })?;

            // Reconstruct the edges vector since it may be out of order.
            // We only store the start indices, since the end is the next's start.
            let mut new_indices: IndexVec<SerializedDepNodeIndex, u32> =
                IndexVec::from_elem_n(0u32, node_count);
            e.emit_struct_field("edge_list_data", 2, |e| {
                e.emit_seq(edge_count, |e| {
                    let mut pos: u32 = 0;
                    for (new_index, edges) in self.edge_map().enumerate() {
                        // Reconstruct the edges vector since it may be out of order.
                        // We only store the end indices, since the start can be reconstructed.
                        for &edge in edges {
                            let edge = remap[edge].unwrap();
                            e.emit_seq_elt(pos as usize, |e| edge.encode(e))?;
                            pos += 1;
                        }
                        new_indices[SerializedDepNodeIndex::new(new_index)] = pos;
                    }
                    debug_assert_eq!(pos as usize, edge_count);
                    Ok(())
                })
            })?;
            e.emit_struct_field("edge_list_ends", 3, |e| new_indices.encode(e))?;
            Ok(())
        })
    }
}

// Pre-allocate the dep node structures. We over-allocate a little so
// that we hopefully don't have to re-allocate during this compilation
// session. The over-allocation for new nodes is 2% plus a small
// constant to account for the fact that in very small crates 2% might
// not be enough. The allocation for red and green node data doesn't
// include a constant, as we don't want to allocate anything for these
// structures during full incremental builds, where they aren't used.
//
// These estimates are based on the distribution of node and edge counts
// seen in rustc-perf benchmarks, adjusted somewhat to account for the
// fact that these benchmarks aren't perfectly representative.
fn node_count_estimate(prev_graph_node_count: usize) -> usize {
    (2 * prev_graph_node_count) / 100 + 200
}

fn edge_count_estimate(prev_graph_node_count: usize) -> usize {
    let average_edges_per_node_estimate = 6;
    average_edges_per_node_estimate * (200 + (prev_graph_node_count * 30) / 100)
}

impl<D: Decoder, K: DepKind + Decodable<D>> Decodable<D> for SerializedDepGraph<K> {
    fn decode(d: &mut D) -> Result<SerializedDepGraph<K>, D::Error> {
        d.read_struct("SerializedDepGraph", 4, |d| {
            let nodes: IndexVec<SerializedDepNodeIndex, DepNode<K>> =
                d.read_struct_field("nodes", 0, |d| {
                    d.read_seq(|d, len| {
                        let mut nodes = IndexVec::with_capacity(len);
                        for i in 0..len {
                            let node = d.read_seq_elt(i, Decodable::decode)?;
                            nodes.push(node);
                        }
                        Ok(nodes)
                    })
                })?;
            let fingerprints = d.read_struct_field("fingerprints", 1, |d| {
                d.read_seq(|d, len| {
                    let mut fingerprints = IndexVec::with_capacity(len);
                    for i in 0..len {
                        let fingerprint = d.read_seq_elt(i, Decodable::decode)?;
                        fingerprints.push(fingerprint);
                    }
                    Ok(fingerprints)
                })
            })?;
            let edge_list_data = d.read_struct_field("edge_list_data", 2, |d| {
                d.read_seq(|d, len| {
                    let mut edges = Vec::with_capacity(len);
                    for i in 0..len {
                        let edge = d.read_seq_elt(i, Decodable::decode)?;
                        edges.push(edge);
                    }
                    Ok(edges)
                })
            })?;
            let edge_list_indices = d.read_struct_field("edge_list_ends", 3, |d| {
                d.read_seq(|d, len| {
                    let mut indices = IndexVec::with_capacity(len);
                    let mut start: u32 = 0;
                    for i in 0..len {
                        let end: u32 = d.read_seq_elt(i, Decodable::decode)?;
                        indices.push(ColorAndOffset::unknown(start, end));
                        start = end;
                    }
                    Ok(indices)
                })
            })?;

            Ok(SerializedDepGraph { nodes, fingerprints, edge_list_indices, edge_list_data })
        })
    }
}
