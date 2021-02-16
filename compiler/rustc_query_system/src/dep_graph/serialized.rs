//! The data that we will serialize and deserialize.

use super::query::DepGraphQuery;
use super::{DepKind, DepNode};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::{Idx, IndexVec};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use std::convert::TryInto;

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

impl SerializedDepNodeIndex {
    pub(super) fn rejuvenate(self) -> DepNodeIndex {
        DepNodeIndex::new(self.index())
    }
}

#[derive(Copy, Clone, Encodable, Decodable)]
struct ColorAndOffset(u32);

impl std::fmt::Debug for ColorAndOffset {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("ColorAndOffset")
            .field("color", &self.color())
            .field("offset", &self.offset())
            .finish()
    }
}

impl ColorAndOffset {
    fn unknown(offset: u32) -> ColorAndOffset {
        debug_assert_eq!(offset & TAG_MASK, 0);
        ColorAndOffset(offset | TAG_UNKNOWN)
    }

    fn new(color: DepNodeColor, offset: usize) -> ColorAndOffset {
        let offset: u32 = offset.try_into().unwrap();
        debug_assert_eq!(offset & TAG_MASK, 0);
        ColorAndOffset(offset | color.tag())
    }

    fn set_color(&mut self, color: DepNodeColor) {
        let offset = self.0 & OFFSET_MASK;
        self.0 = color.tag() | offset;
    }

    fn offset(self) -> u32 {
        self.0 & OFFSET_MASK
    }

    fn color(self) -> Option<DepNodeColor> {
        let tag = self.0 & TAG_MASK;
        match tag {
            TAG_NEW => Some(DepNodeColor::New),
            TAG_RED => Some(DepNodeColor::Red),
            TAG_GREEN => Some(DepNodeColor::Green),
            TAG_UNKNOWN => None,
            _ => panic!(),
        }
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
    edge_list_indices: IndexVec<SerializedDepNodeIndex, (ColorAndOffset, u32)>,
    /// A flattened list of all edge targets in the graph. Edge sources are
    /// implicit in edge_list_indices.
    edge_list_data: Vec<DepNodeIndex>,
    /// Reverse map for `nodes`. It is computed on the fly at decoding time.
    index: FxHashMap<DepNode<K>, SerializedDepNodeIndex>,
    /// Index of the last serialized node.
    serialized_node_count: SerializedDepNodeIndex,
}

impl<K: DepKind> Default for SerializedDepGraph<K> {
    fn default() -> Self {
        Self {
            nodes: IndexVec::new(),
            fingerprints: IndexVec::new(),
            edge_list_indices: IndexVec::new(),
            edge_list_data: Vec::new(),
            index: FxHashMap::default(),
            serialized_node_count: SerializedDepNodeIndex::new(0),
        }
    }
}

impl<K: DepKind> SerializedDepGraph<K> {
    fn intern_new_node(
        &mut self,
        node: DepNode<K>,
        deps: &[DepNodeIndex],
        fingerprint: Fingerprint,
    ) -> DepNodeIndex {
        let index = self.nodes.push(node);
        debug!("intern_new: {:?} {:?}", index, node);
        let _index = self.fingerprints.push(fingerprint);
        debug_assert_eq!(index, _index);
        let (start, end) = self.insert_deps(deps);
        let _index = self
            .edge_list_indices
            .push((ColorAndOffset::new(DepNodeColor::New, start), end.try_into().unwrap()));
        debug_assert_eq!(index, _index);
        let _o = self.index.insert(node, index);
        debug_assert_eq!(_o, None);
        index.rejuvenate()
    }

    fn insert_deps(&mut self, deps: &[DepNodeIndex]) -> (usize, usize) {
        let start = self.edge_list_data.len();
        self.edge_list_data.extend(deps.iter().copied());
        let end = self.edge_list_data.len();
        (start, end)
    }

    fn update_deps(
        &mut self,
        index: SerializedDepNodeIndex,
        color: DepNodeColor,
        deps: &[DepNodeIndex],
    ) {
        let (start, _) = self.edge_list_indices[index];
        debug_assert_eq!(start.color(), None);
        let (start, end) = self.insert_deps(deps);
        debug!("intern_color: {:?} => {:?}", index, color);
        let start = ColorAndOffset::new(color, start);
        self.edge_list_indices[index] = (start, end.try_into().unwrap());
    }

    pub(crate) fn intern_dark_green_node(&mut self, index: SerializedDepNodeIndex) -> DepNodeIndex {
        debug!("intern_drak_green: {:?}", index);
        debug_assert_eq!(self.edge_list_indices[index].0.color(), None);
        self.edge_list_indices[index].0.set_color(DepNodeColor::Green);
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

        if let Some(&prev_index) = self.index.get(&node) {
            if let Some(_) = self.color(prev_index) {
                return prev_index.rejuvenate();
            }

            // Determine the color and index of the new `DepNode`.
            let color = if let Some(fingerprint) = fingerprint {
                if fingerprint == self.fingerprints[prev_index] {
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
                    self.fingerprints[prev_index] = fingerprint;
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
                self.fingerprints[prev_index] = Fingerprint::ZERO;
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
    pub(crate) fn edge_targets_from_serialized(
        &self,
        source: SerializedDepNodeIndex,
    ) -> impl Iterator<Item = SerializedDepNodeIndex> + '_ {
        let (start, end) = self.edge_list_indices[source];
        debug_assert_eq!(start.color(), None);
        let start = start.offset() as usize;
        let end = end as usize;
        self.edge_list_data[start..end].iter().map(|i| SerializedDepNodeIndex::new(i.index()))
    }

    #[inline]
    pub(crate) fn edge_targets_from(&self, source: DepNodeIndex) -> &[DepNodeIndex] {
        let (start, end) = self.edge_list_indices[SerializedDepNodeIndex::new(source.index())];
        let start = start.offset() as usize;
        let end = end as usize;
        &self.edge_list_data[start..end]
    }

    #[inline]
    pub(crate) fn index_to_node(&self, dep_node_index: SerializedDepNodeIndex) -> DepNode<K> {
        self.nodes[dep_node_index]
    }

    #[inline]
    pub(crate) fn dep_node_of(&self, dep_node_index: DepNodeIndex) -> DepNode<K> {
        self.nodes[SerializedDepNodeIndex::new(dep_node_index.index())]
    }

    #[inline]
    pub(crate) fn node_to_index_opt(
        &self,
        dep_node: &DepNode<K>,
    ) -> Option<SerializedDepNodeIndex> {
        let idx = *self.index.get(dep_node)?;
        if idx >= self.serialized_node_count { None } else { Some(idx) }
    }

    #[inline]
    pub(crate) fn dep_node_index_of_opt(&self, dep_node: &DepNode<K>) -> Option<DepNodeIndex> {
        let index = *self.index.get(dep_node)?;
        // Return none if the node has not been coloured yet.
        let _ = self.edge_list_indices[index].0.color()?;
        debug!(
            "dep_node_index_of_opt: dep_node={:?} index={:?} indices={:?}",
            dep_node, index, self.edge_list_indices[index]
        );
        Some(index.rejuvenate())
    }

    #[inline]
    pub(crate) fn color(&self, index: SerializedDepNodeIndex) -> Option<DepNodeColor> {
        self.edge_list_indices[index].0.color()
    }

    #[inline]
    pub(crate) fn fingerprint_of(&self, dep_node_index: DepNodeIndex) -> Fingerprint {
        self.fingerprints[SerializedDepNodeIndex::new(dep_node_index.index())]
    }

    #[inline]
    pub(crate) fn serialized_indices(&self) -> impl Iterator<Item = SerializedDepNodeIndex> + '_ {
        (0..self.serialized_node_count.index()).map(SerializedDepNodeIndex::new)
    }

    #[inline]
    pub(crate) fn indices(&self) -> impl Iterator<Item = DepNodeIndex> + '_ {
        self.edge_list_indices.iter_enumerated().filter_map(|(i, (s, _))| {
            // Return none if the node has not been coloured yet.
            let _ = s.color()?;
            Some(i.rejuvenate())
        })
    }

    #[inline]
    pub(crate) fn serialized_node_count(&self) -> usize {
        self.serialized_node_count.index()
    }

    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        self.edge_list_indices.iter().filter_map(|(s, _)| s.color()).count()
    }

    #[inline]
    pub(crate) fn edge_count(&self) -> usize {
        self.edge_list_indices
            .iter()
            .filter_map(|(s, e)| {
                s.color()?;
                Some((e - s.offset()) as usize)
            })
            .sum()
    }

    pub(crate) fn query(&self) -> DepGraphQuery<K> {
        let nodes: Vec<_> = self
            .nodes
            .iter_enumerated()
            .filter_map(|(i, n)| {
                let _ = self.edge_list_indices[i].0.color()?;
                Some(*n)
            })
            .collect();
        let edge_list_indices: Vec<_> = self
            .edge_list_indices
            .iter()
            .filter_map(|(s, e)| {
                s.color()?;
                Some((s.offset() as usize, *e as usize))
            })
            .collect();
        let edge_list_data: Vec<_> = self.edge_list_data.iter().map(|i| i.index()).collect();
        debug_assert_eq!(nodes.len(), edge_list_indices.len());

        DepGraphQuery::new(&nodes[..], &edge_list_indices[..], &edge_list_data[..])
    }

    pub(crate) fn compression_map(&self) -> IndexVec<DepNodeIndex, Option<SerializedDepNodeIndex>> {
        let mut new_index = SerializedDepNodeIndex::new(0);
        let mut remap = IndexVec::from_elem_n(None, self.nodes.len());
        for index in self.indices() {
            debug_assert!(new_index.index() <= index.index());
            remap[index] = Some(new_index);
            new_index.increment_by(1);
        }
        remap
    }
}

impl<E: Encoder, K: DepKind + Encodable<E>> Encodable<E> for SerializedDepGraph<K> {
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        let remap = self.compression_map();

        let (nodes, fingerprints) = {
            let mut nodes = self.nodes.clone();
            let mut fingerprints = self.fingerprints.clone();
            let mut new_index = SerializedDepNodeIndex::new(0);

            for index in self.indices() {
                debug_assert!(new_index.index() <= index.index());

                // Back-copy the nodes and fingerprints.
                let index = SerializedDepNodeIndex::new(index.index());
                nodes[new_index] = self.nodes[index];
                fingerprints[new_index] = self.fingerprints[index];

                new_index.increment_by(1);
            }
            nodes.truncate(new_index.index());
            fingerprints.truncate(new_index.index());

            (nodes, fingerprints)
        };

        let (new_indices, new_edges) = {
            let mut new_indices: IndexVec<SerializedDepNodeIndex, u32> =
                IndexVec::with_capacity(self.nodes.len());
            let mut new_edges: Vec<SerializedDepNodeIndex> =
                Vec::with_capacity(self.edge_list_data.len());

            for (index, (start, end)) in self.edge_list_indices.iter_enumerated() {
                match start.color() {
                    // This node does not exist in this session. Skip it.
                    None => continue,
                    Some(_) => {}
                }

                let new_index = new_indices.push(new_edges.len().try_into().unwrap());
                debug_assert_eq!(remap[index.rejuvenate()], Some(new_index));

                // Reconstruct the edges vector since it may be out of order.
                // We only store the start indices, since the end is the next's start.
                let start = start.offset() as usize;
                let end = *end as usize;
                new_edges.extend(self.edge_list_data[start..end].iter().map(|i| {
                    remap[*i]
                        .unwrap_or_else(|| panic!("Unknown remap for {:?} while {:?}", *i, index))
                }));
            }

            (new_indices, new_edges)
        };

        let mut index = FxHashMap::default();
        for (idx, &dep_node) in nodes.iter_enumerated() {
            debug!("DECODE index={:?} node={:?}", idx, dep_node);
            let _o = index.insert(dep_node, idx);
            debug_assert_eq!(_o, None);
        }
        let _ = index;

        e.emit_struct("SerializedDepGraph", 4, |e| {
            e.emit_struct_field("nodes", 0, |e| nodes.encode(e))?;
            e.emit_struct_field("fingerprints", 1, |e| fingerprints.encode(e))?;
            e.emit_struct_field("edge_list_indices", 2, |e| new_indices.encode(e))?;
            e.emit_struct_field("edge_list_data", 3, |e| new_edges.encode(e))?;
            Ok(())
        })
    }
}

impl<D: Decoder, K: DepKind + Decodable<D>> Decodable<D> for SerializedDepGraph<K> {
    fn decode(d: &mut D) -> Result<SerializedDepGraph<K>, D::Error> {
        d.read_struct("SerializedDepGraph", 4, |d| {
            let nodes: IndexVec<SerializedDepNodeIndex, DepNode<K>> =
                d.read_struct_field("nodes", 0, Decodable::decode)?;
            let fingerprints: IndexVec<SerializedDepNodeIndex, Fingerprint> =
                d.read_struct_field("fingerprints", 1, Decodable::decode)?;
            let mut edge_list_indices: IndexVec<SerializedDepNodeIndex, u32> =
                d.read_struct_field("edge_list_indices", 2, Decodable::decode)?;
            let edge_list_data: Vec<DepNodeIndex> =
                d.read_struct_field("edge_list_data", 3, Decodable::decode)?;

            edge_list_indices.push(edge_list_data.len().try_into().unwrap());
            let edge_list_indices = IndexVec::from_fn_n(
                |i| (ColorAndOffset::unknown(edge_list_indices[i]), edge_list_indices[i + 1]),
                edge_list_indices.len() - 1,
            );

            let mut index = FxHashMap::default();
            for (idx, &dep_node) in nodes.iter_enumerated() {
                debug!("DECODE index={:?} node={:?}", idx, dep_node);
                let _o = index.insert(dep_node, idx);
                debug_assert_eq!(_o, None);
            }
            let serialized_node_count = nodes.next_index();

            Ok(SerializedDepGraph {
                nodes,
                index,
                fingerprints,
                edge_list_indices,
                edge_list_data,
                serialized_node_count,
            })
        })
    }
}
