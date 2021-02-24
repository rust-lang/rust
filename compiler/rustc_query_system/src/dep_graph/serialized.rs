//! The data that we will serialize and deserialize.

use super::query::DepGraphQuery;
use super::{DepKind, DepNode};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::{AtomicU64, Ordering};
use rustc_index::vec::{Idx, IndexArray, IndexVec};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use std::convert::TryInto;
use std::ops::Range;

#[derive(Debug, PartialEq, Eq)]
pub enum DepNodeColor {
    Green,
    Red,
    New,
}

const TAG_UNKNOWN: u64 = 0;
const TAG_GREEN: u64 = 1 << 62;
const TAG_RED: u64 = 2 << 62;
const TAG_NEW: u64 = 3 << 62;
const TAG_LIFTED: u64 = 1 << 31;
const TAG_MASK: u64 = TAG_UNKNOWN | TAG_GREEN | TAG_RED | TAG_NEW | TAG_LIFTED;
const OFFSET_MASK: u64 = !TAG_MASK;

impl DepNodeColor {
    const fn tag(self) -> u64 {
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

struct ColorAndOffset(AtomicU64);

impl std::fmt::Debug for ColorAndOffset {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (lifted, range) = self.range();
        fmt.debug_struct("ColorAndOffset")
            .field("color", &self.color())
            .field("lifted", &lifted)
            .field("range", &range)
            .finish()
    }
}

impl ColorAndOffset {
    fn unknown(start: u32, end: u32) -> ColorAndOffset {
        let val = (start as u64) << 32 | (end as u64);
        debug_assert_eq!(val & TAG_MASK, 0);
        ColorAndOffset(AtomicU64::new(val))
    }

    #[allow(dead_code)]
    fn new(color: DepNodeColor, range: Range<usize>) -> ColorAndOffset {
        let start: u32 = range.start.try_into().unwrap();
        let end: u32 = range.end.try_into().unwrap();
        let val = (start as u64) << 32 | (end as u64);
        debug_assert_eq!(val & TAG_MASK, 0);
        let val = val | color.tag();
        ColorAndOffset(AtomicU64::new(val))
    }

    fn set_lifted(&self, color: DepNodeColor, range: Range<usize>) {
        let start: u32 = range.start.try_into().unwrap();
        let end: u32 = range.end.try_into().unwrap();
        let val = (start as u64) << 32 | (end as u64);
        debug_assert_eq!(val & TAG_MASK, 0);
        let val = val | color.tag() | TAG_LIFTED;
        self.0.store(val, Ordering::Release)
    }

    fn set_color(&self, color: DepNodeColor) {
        self.0.fetch_or(color.tag(), Ordering::SeqCst);
    }

    fn color(&self) -> Option<DepNodeColor> {
        let tag = self.0.load(Ordering::Acquire) & TAG_MASK & !TAG_LIFTED;
        match tag {
            TAG_NEW => Some(DepNodeColor::New),
            TAG_RED => Some(DepNodeColor::Red),
            TAG_GREEN => Some(DepNodeColor::Green),
            TAG_UNKNOWN => None,
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    fn range(&self) -> (bool, Range<usize>) {
        let val = self.0.load(Ordering::Acquire);
        let lifted = val & TAG_LIFTED != 0;
        let val = val & OFFSET_MASK;
        let start = (val >> 32) as usize;
        let end = val as u32 as usize;
        (lifted, start..end)
    }
}

fn shrink_range(range: Range<usize>) -> Range<u32> {
    range.start.try_into().unwrap()..range.end.try_into().unwrap()
}

/// Data for use when recompiling the **previous crate**.
#[derive(Debug)]
pub struct SerializedDepGraph<K: DepKind> {
    /// The set of all DepNodes in the graph
    nodes: IndexArray<SerializedDepNodeIndex, DepNode<K>>,
    /// The set of all Fingerprints in the graph. Each Fingerprint corresponds to
    /// the DepNode at the same index in the nodes vector.
    // This field must only be read or modified when holding a lock to the CurrentDepGraph.
    fingerprints: IndexArray<SerializedDepNodeIndex, Fingerprint>,
    /// For each DepNode, stores the list of edges originating from that
    /// DepNode. Encoded as a [start, end) pair indexing into edge_list_data,
    /// which holds the actual DepNodeIndices of the target nodes.
    edge_list_indices: IndexArray<SerializedDepNodeIndex, ColorAndOffset>,
    /// A flattened list of all edge targets in the graph. Edge sources are
    /// implicit in edge_list_indices.
    edge_list_data: Vec<DepNodeIndex>,
}

/// Data for use when recompiling the **current crate**.
#[derive(Debug)]
pub struct CurrentDepGraph<K: DepKind> {
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
            nodes: IndexArray::new(),
            fingerprints: IndexArray::new(),
            edge_list_indices: IndexArray::new(),
            edge_list_data: Vec::new(),
        }
    }
}

impl<K: DepKind> SerializedDepGraph<K> {
    pub(crate) fn intern_dark_green_node(&self, index: SerializedDepNodeIndex) -> DepNodeIndex {
        debug!("intern_drak_green: {:?}", index);
        debug_assert_eq!(self.edge_list_indices[index].color(), None);
        self.edge_list_indices[index].set_color(DepNodeColor::Green);
        debug!("intern_color: {:?} => Green", index);
        index.rejuvenate()
    }

    #[inline]
    pub(crate) fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> DepNode<K> {
        self.nodes[dep_node_index]
    }

    #[inline]
    pub(crate) fn color(&self, index: SerializedDepNodeIndex) -> Option<DepNodeColor> {
        self.edge_list_indices[index].color()
    }

    #[inline]
    pub(crate) fn color_or_edges(
        &self,
        source: SerializedDepNodeIndex,
    ) -> Result<DepNodeColor, &'static [SerializedDepNodeIndex]> {
        let range = &self.edge_list_indices[source];
        if let Some(color) = range.color() {
            return Ok(color);
        }
        // The node has not been colored, so the dependencies have not been lifted to point to the
        // new nodes vector.
        let (_lifted, range) = range.range();
        debug_assert!(!_lifted);
        let edges = &self.edge_list_data[range];
        debug_assert_eq!(
            std::mem::size_of::<DepNodeIndex>(),
            std::mem::size_of::<SerializedDepNodeIndex>()
        );
        // SAFETY: 1. self.edge_list_data is never modified.
        // 2. SerializedDepNodeIndex and DepNodeIndex have the same binary representation.
        let edges = unsafe { std::mem::transmute::<&[_], &[_]>(edges) };
        Err(edges)
    }

    #[inline]
    fn as_serialized(&self, index: DepNodeIndex) -> Result<SerializedDepNodeIndex, SplitIndex> {
        let index = index.index();
        let count = self.nodes.len();
        if index < count {
            Ok(SerializedDepNodeIndex::new(index))
        } else {
            Err(SplitIndex::new(index - count))
        }
    }

    #[inline]
    fn from_split(&self, index: SplitIndex) -> DepNodeIndex {
        DepNodeIndex::new(self.nodes.len() + index.index())
    }

    #[inline]
    pub(crate) fn serialized_indices(&self) -> impl Iterator<Item = SerializedDepNodeIndex> {
        self.nodes.indices()
    }

    #[inline]
    fn live_serialized_indices(&self) -> impl Iterator<Item = SerializedDepNodeIndex> + '_ {
        self.edge_list_indices.iter_enumerated().filter_map(|(i, range)| {
            // Return none if the node has not been coloured yet.
            let _ = range.color()?;
            Some(i)
        })
    }
}

impl<K: DepKind> CurrentDepGraph<K> {
    pub(crate) fn new(serialized: &SerializedDepGraph<K>) -> Self {
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
            nodes: IndexVec::with_capacity(nodes),
            fingerprints: IndexVec::with_capacity(nodes),
            edge_list_indices: IndexVec::with_capacity(nodes),
            edge_list_data: Vec::with_capacity(edges),
            index,
        }
    }

    fn intern_new_node(
        &mut self,
        serialized: &SerializedDepGraph<K>,
        node: DepNode<K>,
        deps: &[DepNodeIndex],
        fingerprint: Fingerprint,
    ) -> DepNodeIndex {
        let index = self.nodes.push(node);
        debug!("intern_new: {:?} {:?}", serialized.from_split(index), node);
        let _index = self.fingerprints.push(fingerprint);
        debug_assert_eq!(index, _index);
        let range = self.insert_deps(deps);
        let _index = self.edge_list_indices.push(shrink_range(range));
        debug_assert_eq!(index, _index);
        let index = serialized.from_split(index);
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
        serialized: &SerializedDepGraph<K>,
        index: SerializedDepNodeIndex,
        color: DepNodeColor,
        deps: &[DepNodeIndex],
    ) {
        debug!("intern_color: {:?} => {:?}", index, color);
        let range = &serialized.edge_list_indices[index];
        debug_assert_eq!(range.color(), None);
        let range = self.insert_deps(deps);
        serialized.edge_list_indices[index].set_lifted(color, range);
    }

    pub(crate) fn intern_anon_node(
        &mut self,
        serialized: &SerializedDepGraph<K>,
        node: DepNode<K>,
        deps: &[DepNodeIndex],
    ) -> DepNodeIndex {
        self.dep_node_index_of_opt(serialized, &node)
            .unwrap_or_else(|| self.intern_new_node(serialized, node, deps, Fingerprint::ZERO))
    }

    pub(crate) fn intern_task_node(
        &mut self,
        serialized: &SerializedDepGraph<K>,
        node: DepNode<K>,
        deps: &[DepNodeIndex],
        fingerprint: Option<Fingerprint>,
        print_status: bool,
    ) -> DepNodeIndex {
        let print_status = cfg!(debug_assertions) && print_status;

        if let Some(&existing) = self.index.get(&node) {
            let prev_index = serialized
                .as_serialized(existing)
                .unwrap_or_else(|_| panic!("Node {:?} is being interned multiple times.", node));
            match serialized.color(prev_index) {
                Some(DepNodeColor::Red) | Some(DepNodeColor::New) => {
                    panic!("Node {:?} is being interned multiple times.", node)
                }

                // This can happen when trying to compute the result of green queries.
                Some(DepNodeColor::Green) => return existing,

                None => {}
            }

            // Determine the color and index of the new `DepNode`.
            let color = if let Some(fingerprint) = fingerprint {
                if fingerprint == serialized.fingerprints[prev_index] {
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
                    // SAFETY: `serialized.fingerprints` is only read or mutated when holding a
                    // lock to CurrentDepGraph.
                    unsafe {
                        *(&serialized.fingerprints[prev_index] as *const _ as *mut _) = fingerprint
                    };
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
                // SAFETY: `serialized.fingerprints` is only read or mutated when holding a
                // lock to CurrentDepGraph.
                unsafe {
                    *(&serialized.fingerprints[prev_index] as *const _ as *mut _) =
                        Fingerprint::ZERO
                };
                DepNodeColor::Red
            };

            self.update_deps(serialized, prev_index, color, deps);
            prev_index.rejuvenate()
        } else {
            if print_status {
                eprintln!("[task::new] {:?}", node);
            }

            // This is a new node: it didn't exist in the previous compilation session.
            self.intern_new_node(serialized, node, deps, fingerprint.unwrap_or(Fingerprint::ZERO))
        }
    }

    #[inline]
    pub(crate) fn node_to_index_opt(
        &self,
        serialized: &SerializedDepGraph<K>,
        dep_node: &DepNode<K>,
    ) -> Option<SerializedDepNodeIndex> {
        let idx = *self.index.get(dep_node)?;
        serialized.as_serialized(idx).ok()
    }

    #[inline]
    fn serialized_edges<'a>(
        &'a self,
        serialized: &'a SerializedDepGraph<K>,
        source: SerializedDepNodeIndex,
    ) -> &[DepNodeIndex] {
        let range = &serialized.edge_list_indices[source];
        let (lifted, range) = range.range();
        if lifted { &self.edge_list_data[range] } else { &serialized.edge_list_data[range] }
    }

    #[inline]
    fn new_edges(&self, source: SplitIndex) -> &[DepNodeIndex] {
        let range = &self.edge_list_indices[source];
        let start = range.start as usize;
        let end = range.end as usize;
        &self.edge_list_data[start..end]
    }

    #[inline]
    pub(crate) fn edge_targets_from<'a>(
        &'a self,
        serialized: &'a SerializedDepGraph<K>,
        source: DepNodeIndex,
    ) -> &[DepNodeIndex] {
        match serialized.as_serialized(source) {
            Ok(source) => self.serialized_edges(serialized, source),
            Err(source) => self.new_edges(source),
        }
    }

    #[inline]
    pub(crate) fn index_to_node(
        &self,
        serialized: &SerializedDepGraph<K>,
        dep_node_index: DepNodeIndex,
    ) -> DepNode<K> {
        match serialized.as_serialized(dep_node_index) {
            Ok(sni) => serialized.nodes[sni],
            Err(new) => self.nodes[new],
        }
    }

    #[inline]
    pub(crate) fn dep_node_index_of_opt(
        &self,
        serialized: &SerializedDepGraph<K>,
        dep_node: &DepNode<K>,
    ) -> Option<DepNodeIndex> {
        let index = *self.index.get(dep_node)?;
        if let Ok(prev) = serialized.as_serialized(index) {
            // Return none if the node has not been coloured yet.
            serialized.edge_list_indices[prev].color()?;
        }
        Some(index)
    }

    #[inline]
    pub(crate) fn fingerprint_of(
        &self,
        serialized: &SerializedDepGraph<K>,
        dep_node_index: DepNodeIndex,
    ) -> Fingerprint {
        match serialized.as_serialized(dep_node_index) {
            Ok(sni) => serialized.fingerprints[sni],
            Err(split) => self.fingerprints[split],
        }
    }

    #[inline]
    fn new_indices<'a>(
        &self,
        serialized: &'a SerializedDepGraph<K>,
    ) -> impl Iterator<Item = DepNodeIndex> + 'a {
        self.nodes.indices().map(move |i| serialized.from_split(i))
    }

    #[inline]
    pub(crate) fn live_indices<'a>(
        &self,
        serialized: &'a SerializedDepGraph<K>,
    ) -> impl Iterator<Item = DepNodeIndex> + 'a {
        // New indices are always live.
        serialized
            .live_serialized_indices()
            .map(SerializedDepNodeIndex::rejuvenate)
            .chain(self.new_indices(serialized))
    }

    #[inline]
    pub(crate) fn node_count(&self, serialized: &SerializedDepGraph<K>) -> usize {
        self.live_indices(serialized).count()
    }

    #[inline]
    fn edge_map<'a>(
        &'a self,
        serialized: &'a SerializedDepGraph<K>,
    ) -> impl Iterator<Item = &'a [DepNodeIndex]> + 'a {
        let serialized_edges = serialized
            .live_serialized_indices()
            .map(move |index| self.serialized_edges(serialized, index));
        let new_edges = self.edge_list_indices.iter().map(move |range| {
            let start = range.start as usize;
            let end = range.end as usize;
            &self.edge_list_data[start..end]
        });
        serialized_edges.chain(new_edges)
    }

    #[inline]
    pub(crate) fn edge_count(&self, serialized: &SerializedDepGraph<K>) -> usize {
        self.edge_map(serialized).flatten().count()
    }

    pub(crate) fn query(&self, serialized: &SerializedDepGraph<K>) -> DepGraphQuery<K> {
        let node_count = self.node_count(serialized);
        let edge_count = self.edge_count(serialized);

        let mut nodes = Vec::with_capacity(node_count);
        nodes.extend(self.live_indices(serialized).map(|i| self.index_to_node(serialized, i)));

        let mut edge_list_indices = Vec::with_capacity(node_count);
        let mut edge_list_data = Vec::with_capacity(edge_count);
        for edges in self.edge_map(serialized) {
            let start = edge_list_data.len();
            edge_list_data.extend(edges.iter().map(|i| i.index() as usize));
            let end = edge_list_data.len();
            edge_list_indices.push((start, end))
        }

        debug_assert_eq!(nodes.len(), edge_list_indices.len());
        DepGraphQuery::new(&nodes[..], &edge_list_indices[..], &edge_list_data[..])
    }

    pub(crate) fn compression_map(
        &self,
        serialized: &SerializedDepGraph<K>,
    ) -> IndexVec<DepNodeIndex, Option<SerializedDepNodeIndex>> {
        let mut new_index = SerializedDepNodeIndex::new(0);
        let mut remap = IndexVec::from_elem_n(None, serialized.nodes.len() + self.nodes.len());
        for index in self.live_indices(serialized) {
            debug_assert!(new_index.index() <= index.index());
            remap[index] = Some(new_index);
            new_index.increment_by(1);
        }
        remap
    }
}

impl<K: DepKind> CurrentDepGraph<K> {
    pub(crate) fn encode<E: Encoder>(
        &self,
        serialized: &SerializedDepGraph<K>,
        e: &mut E,
    ) -> Result<(), E::Error>
    where
        K: Encodable<E>,
    {
        let remap = self.compression_map(serialized);
        let live_indices = || remap.iter_enumerated().filter_map(|(s, &n)| Some((s, n?)));
        let node_count = live_indices().count();
        let edge_count = self.edge_count(serialized);

        e.emit_struct("SerializedDepGraph", 4, |e| {
            e.emit_struct_field("nodes", 0, |e| {
                e.emit_seq(node_count, |e| {
                    for (index, new_index) in live_indices() {
                        let node = self.index_to_node(serialized, index);
                        e.emit_seq_elt(new_index.index(), |e| node.encode(e))?;
                    }
                    Ok(())
                })
            })?;
            e.emit_struct_field("fingerprints", 1, |e| {
                e.emit_seq(node_count, |e| {
                    for (index, new_index) in live_indices() {
                        let node = self.fingerprint_of(serialized, index);
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
                    for (new_index, edges) in self.edge_map(serialized).enumerate() {
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
            let nodes = d.read_struct_field("nodes", 0, |d| {
                d.read_seq(|d, len| {
                    let mut nodes = Vec::with_capacity(len);
                    for i in 0..len {
                        let node = d.read_seq_elt(i, Decodable::decode)?;
                        nodes.push(node);
                    }
                    Ok(nodes)
                })
            })?;
            let fingerprints = d.read_struct_field("fingerprints", 1, |d| {
                d.read_seq(|d, len| {
                    let mut fingerprints = Vec::with_capacity(len);
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
                    let mut indices = Vec::with_capacity(len);
                    let mut start: u32 = 0;
                    for i in 0..len {
                        let end: u32 = d.read_seq_elt(i, Decodable::decode)?;
                        indices.push(ColorAndOffset::unknown(start, end));
                        start = end;
                    }
                    Ok(indices)
                })
            })?;

            Ok(SerializedDepGraph {
                nodes: IndexArray::from_vec(nodes),
                fingerprints: IndexArray::from_vec(fingerprints),
                edge_list_indices: IndexArray::from_vec(edge_list_indices),
                edge_list_data,
            })
        })
    }
}
