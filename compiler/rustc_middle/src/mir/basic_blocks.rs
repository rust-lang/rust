use std::sync::{Arc, OnceLock};

use rustc_data_structures::graph;
use rustc_data_structures::graph::dominators::{Dominators, dominators};
use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_index::{IndexSlice, IndexVec};
use rustc_macros::{StableHash, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use smallvec::SmallVec;

use crate::mir::traversal::Postorder;
use crate::mir::{BasicBlock, BasicBlockData, START_BLOCK};

#[derive(Clone, TyEncodable, TyDecodable, Debug, StableHash, TypeFoldable, TypeVisitable)]
pub struct BasicBlocks<'tcx> {
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    /// Use an `Arc` so we can share the cache when we clone the MIR body, as borrowck does.
    cache: Arc<CfgCache>,
}

// Typically 95%+ of basic blocks have 4 or fewer predecessors.
pub type Predecessors = IndexVec<BasicBlock, SmallVec<[BasicBlock; 4]>>;

#[derive(Clone, Default, Debug)]
pub struct CfgCache {
    predecessors: OnceLock<Predecessors>,
    reverse_postorder: OnceLock<IndexVec<BasicBlock, BasicBlock>>,
    dominators: OnceLock<Dominators<BasicBlock>>,
}

impl CfgCache {
    #[inline]
    pub fn dominators(&self, bbs: &'_ BasicBlocks<'_>) -> &Dominators<BasicBlock> {
        self.dominators.get_or_init(|| dominators(bbs))
    }

    #[inline]
    pub fn reverse_postorder(
        &self,
        bbs: &'_ BasicBlocks<'_>,
    ) -> &IndexSlice<BasicBlock, BasicBlock> {
        self.reverse_postorder.get_or_init(|| {
            let mut rpo: IndexVec<BasicBlock, BasicBlock> =
                Postorder::new(&bbs.basic_blocks, START_BLOCK, None).collect();
            rpo.raw.reverse();
            rpo
        })
    }

    #[inline]
    pub fn predecessors(&self, bbs: &'_ BasicBlocks<'_>) -> &Predecessors {
        self.predecessors.get_or_init(|| {
            let mut preds = IndexVec::from_elem(SmallVec::new(), &bbs.basic_blocks);
            for (bb, data) in bbs.basic_blocks.iter_enumerated() {
                for succ in data.terminator().successors() {
                    preds[succ].push(bb);
                }
            }
            preds
        })
    }
}

impl<'tcx> BasicBlocks<'tcx> {
    #[inline]
    pub fn new(basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>) -> Self {
        BasicBlocks { basic_blocks, cache: Arc::new(CfgCache::default()) }
    }

    #[inline]
    pub fn cfg_cache(&self) -> Arc<CfgCache> {
        self.cache.clone()
    }

    #[inline]
    pub fn dominators(&self) -> &Dominators<BasicBlock> {
        self.cache.dominators(self)
    }

    /// Returns predecessors for each basic block.
    #[inline]
    pub fn predecessors(&self) -> &Predecessors {
        self.cache.predecessors(self)
    }

    /// Returns basic blocks in a reverse postorder.
    ///
    /// See [`traversal::reverse_postorder`]'s docs to learn what is postorder traversal.
    ///
    /// [`traversal::reverse_postorder`]: crate::mir::traversal::reverse_postorder
    #[inline]
    pub fn reverse_postorder(&self) -> &IndexSlice<BasicBlock, BasicBlock> {
        self.cache.reverse_postorder(self)
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
    #[inline]
    pub fn as_mut_preserves_cfg(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        &mut self.basic_blocks
    }

    /// Invalidates cached information about the CFG.
    ///
    /// You will only ever need this if you have also called [`BasicBlocks::as_mut_preserves_cfg`].
    /// All other methods that allow you to mutate the basic blocks also call this method
    /// themselves, thereby avoiding any risk of accidentally cache invalidation.
    pub fn invalidate_cfg_cache(&mut self) {
        if let Some(cache) = Arc::get_mut(&mut self.cache) {
            // If we only have a single reference to this cache, clear it.
            *cache = CfgCache::default();
        } else {
            // If we have several references to this cache, overwrite the pointer itself so other
            // users can continue to use their (valid) cache.
            self.cache = Arc::new(CfgCache::default());
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
TrivialTypeTraversalImpls! { CfgCache }

impl<S: Encoder> Encodable<S> for CfgCache {
    #[inline]
    fn encode(&self, _s: &mut S) {}
}

impl<D: Decoder> Decodable<D> for CfgCache {
    #[inline]
    fn decode(_: &mut D) -> Self {
        Default::default()
    }
}

impl StableHash for CfgCache {
    #[inline]
    fn stable_hash<Hcx: StableHashCtxt>(&self, _: &mut Hcx, _: &mut StableHasher) {}
}
