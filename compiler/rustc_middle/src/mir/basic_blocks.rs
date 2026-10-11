use std::sync::{Arc, OnceLock};

use rustc_data_structures::graph;
use rustc_data_structures::graph::dominators::{Dominators, dominators};
use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_index::{IndexSlice, IndexVec};
use rustc_macros::{StableHash, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use smallvec::SmallVec;

use crate::mir::traversal::Postorder;
use crate::mir::{BasicBlock, BasicBlockData, DenseLocationMap, START_BLOCK};

#[derive(Clone, TyEncodable, TyDecodable, Debug, StableHash, TypeFoldable, TypeVisitable)]
pub struct BasicBlocks<'tcx> {
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    /// Use an `Arc` so we can share the cache when we clone the MIR body, as borrowck does.
    cache: Arc<Cache>,
    /// See [`BasicBlocks::location_map`]. Unlike the CFG cache this also depends on the
    /// statements, so each `BasicBlocks` keeps its own and only shares the computed map.
    location_map: LocationMapCache,
}

// Typically 95%+ of basic blocks have 4 or fewer predecessors.
type Predecessors = IndexVec<BasicBlock, SmallVec<[BasicBlock; 4]>>;

#[derive(Clone, Default, Debug)]
struct Cache {
    predecessors: OnceLock<Predecessors>,
    reverse_postorder: OnceLock<Vec<BasicBlock>>,
    dominators: OnceLock<Dominators<BasicBlock>>,
}

#[derive(Clone, Default, Debug)]
struct LocationMapCache(OnceLock<Arc<DenseLocationMap>>);

impl Cache {
    fn is_empty(&self) -> bool {
        self.predecessors.get().is_none()
            && self.reverse_postorder.get().is_none()
            && self.dominators.get().is_none()
    }
}

impl<'tcx> BasicBlocks<'tcx> {
    #[inline]
    pub fn new(basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>) -> Self {
        BasicBlocks {
            basic_blocks,
            cache: Arc::new(Cache::default()),
            location_map: Default::default(),
        }
    }

    #[inline]
    pub fn dominators(&self) -> &Dominators<BasicBlock> {
        self.cache.dominators.get_or_init(|| dominators(self))
    }

    /// Returns predecessors for each basic block.
    #[inline]
    pub fn predecessors(&self) -> &Predecessors {
        self.cache.predecessors.get_or_init(|| {
            let mut preds = IndexVec::from_elem(SmallVec::new(), &self.basic_blocks);
            for (bb, data) in self.basic_blocks.iter_enumerated() {
                if let Some(term) = &data.terminator {
                    for succ in term.successors() {
                        preds[succ].push(bb);
                    }
                }
            }
            preds
        })
    }

    /// Returns basic blocks in a reverse postorder.
    ///
    /// See [`traversal::reverse_postorder`]'s docs to learn what is postorder traversal.
    ///
    /// [`traversal::reverse_postorder`]: crate::mir::traversal::reverse_postorder
    #[inline]
    pub fn reverse_postorder(&self) -> &[BasicBlock] {
        self.cache.reverse_postorder.get_or_init(|| {
            let mut rpo: Vec<_> = Postorder::new(&self.basic_blocks, START_BLOCK, None).collect();
            rpo.reverse();
            rpo
        })
    }

    /// Returns the numbering of the body's locations as `PointIndex`es, computed once per body
    /// and shared by everything that keeps per-location data.
    #[inline]
    pub fn location_map(&self) -> &Arc<DenseLocationMap> {
        self.location_map.0.get_or_init(|| Arc::new(DenseLocationMap::new(&self.basic_blocks)))
    }

    /// Returns mutable reference to basic blocks. Invalidates CFG cache.
    #[inline]
    pub fn as_mut(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        self.invalidate_cfg_cache();
        &mut self.basic_blocks
    }

    /// Get mutable access to basic blocks without invalidating the CFG cache.
    ///
    /// By calling this method instead of e.g. [`BasicBlocks::as_mut`] you promise not to change
    /// the CFG. This means that
    ///
    ///  1) The number of basic blocks remains unchanged
    ///  2) The set of successors of each terminator remains unchanged.
    ///  3) For each `TerminatorKind::SwitchInt`, the `targets` remains the same and the terminator
    ///     kind is not changed.
    ///
    /// If any of these conditions cannot be upheld, you should call [`BasicBlocks::invalidate_cfg_cache`].
    ///
    /// Statements may still be added or removed, which renumbers the locations, so the location
    /// map is invalidated.
    #[inline]
    pub fn as_mut_preserves_cfg(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        self.location_map.0.take();
        &mut self.basic_blocks
    }

    /// Invalidates cached information about the CFG.
    ///
    /// You will only ever need this if you have also called [`BasicBlocks::as_mut_preserves_cfg`].
    /// All other methods that allow you to mutate the basic blocks also call this method
    /// themselves, thereby avoiding any risk of accidentally cache invalidation.
    pub fn invalidate_cfg_cache(&mut self) {
        self.location_map.0.take();
        if let Some(cache) = Arc::get_mut(&mut self.cache) {
            // If we only have a single reference to this cache, clear it. Code that builds a body
            // block by block gets here with nothing cached yet, so do not touch it then.
            if !cache.is_empty() {
                *cache = Cache::default();
            }
        } else {
            // If we have several references to this cache, overwrite the pointer itself so other
            // users can continue to use their (valid) cache.
            self.cache = Arc::new(Cache::default());
        }
    }
}

impl<'tcx> std::ops::Deref for BasicBlocks<'tcx> {
    type Target = IndexSlice<BasicBlock, BasicBlockData<'tcx>>;

    #[inline]
    fn deref(&self) -> &IndexSlice<BasicBlock, BasicBlockData<'tcx>> {
        &self.basic_blocks
    }
}

impl<'tcx> graph::DirectedGraph for BasicBlocks<'tcx> {
    type Node = BasicBlock;

    #[inline]
    fn num_nodes(&self) -> usize {
        self.basic_blocks.len()
    }
}

impl<'tcx> graph::StartNode for BasicBlocks<'tcx> {
    #[inline]
    fn start_node(&self) -> Self::Node {
        START_BLOCK
    }
}

impl<'tcx> graph::Successors for BasicBlocks<'tcx> {
    #[inline]
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.basic_blocks[node].terminator().successors()
    }
}

impl<'tcx> graph::Predecessors for BasicBlocks<'tcx> {
    #[inline]
    fn predecessors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.predecessors()[node].iter().copied()
    }
}

// Done here instead of in `structural_impls.rs` because `Cache` is private, as is `basic_blocks`.
TrivialTypeTraversalImpls! { Cache, LocationMapCache }

impl<S: Encoder> Encodable<S> for Cache {
    #[inline]
    fn encode(&self, _s: &mut S) {}
}

impl<D: Decoder> Decodable<D> for Cache {
    #[inline]
    fn decode(_: &mut D) -> Self {
        Default::default()
    }
}

impl StableHash for Cache {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, _: &mut Hcx, _: &mut StableHasher) {}
}

impl<S: Encoder> Encodable<S> for LocationMapCache {
    #[inline]
    fn encode(&self, _s: &mut S) {}
}

impl<D: Decoder> Decodable<D> for LocationMapCache {
    #[inline]
    fn decode(_: &mut D) -> Self {
        Default::default()
    }
}

impl StableHash for LocationMapCache {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, _: &mut Hcx, _: &mut StableHasher) {}
}
