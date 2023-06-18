use crate::mir::traversal::Postorder;
use crate::mir::{BasicBlock, BasicBlockData, Successors, Terminator, TerminatorKind, START_BLOCK};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph;
use rustc_data_structures::graph::dominators::{dominators, Dominators};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::OnceCell;
use rustc_index::{IndexSlice, IndexVec};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use smallvec::SmallVec;

#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable, TypeFoldable, TypeVisitable)]
pub struct BasicBlocks<'tcx> {
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    cache: Cache,
}

// Typically 95%+ of basic blocks have 4 or fewer predecessors.
pub type Predecessors = IndexVec<BasicBlock, SmallVec<[BasicBlock; 4]>>;

pub type SwitchSources = FxHashMap<(BasicBlock, BasicBlock), SmallVec<[Option<u128>; 1]>>;

#[derive(Clone, Default, Debug)]
struct Cache {
    predecessors: OnceCell<Predecessors>,
    switch_sources: OnceCell<SwitchSources>,
    is_cyclic: OnceCell<bool>,
    reverse_postorder: OnceCell<Vec<BasicBlock>>,
    dominators: OnceCell<Dominators<BasicBlock>>,
}

impl<'tcx> BasicBlocks<'tcx> {
    #[inline]
    pub fn new(basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>) -> Self {
        BasicBlocks { basic_blocks, cache: Cache::default() }
    }

    /// Returns true if control-flow graph contains a cycle reachable from the `START_BLOCK`.
    #[inline]
    pub fn is_cfg_cyclic(&self) -> bool {
        *self.cache.is_cyclic.get_or_init(|| graph::is_cyclic(self))
    }

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
    #[inline]
    pub fn reverse_postorder(&self) -> &[BasicBlock] {
        self.cache.reverse_postorder.get_or_init(|| {
            let mut rpo: Vec<_> =
                Postorder::new(&self.basic_blocks, START_BLOCK).map(|(bb, _)| bb).collect();
            rpo.reverse();
            rpo
        })
    }

    /// `switch_sources()[&(target, switch)]` returns a list of switch
    /// values that lead to a `target` block from a `switch` block.
    #[inline]
    pub fn switch_sources(&self) -> &SwitchSources {
        self.cache.switch_sources.get_or_init(|| {
            let mut switch_sources: SwitchSources = FxHashMap::default();
            for (bb, data) in self.basic_blocks.iter_enumerated() {
                if let Some(Terminator {
                    kind: TerminatorKind::SwitchInt { targets, .. }, ..
                }) = &data.terminator
                {
                    for (value, target) in targets.iter() {
                        switch_sources.entry((target, bb)).or_default().push(Some(value));
                    }
                    switch_sources.entry((targets.otherwise(), bb)).or_default().push(None);
                }
            }
            switch_sources
        })
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
        self.cache = Cache::default();
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
}

impl<'tcx> graph::WithNumNodes for BasicBlocks<'tcx> {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.basic_blocks.len()
    }
}

impl<'tcx> graph::WithStartNode for BasicBlocks<'tcx> {
    #[inline]
    fn start_node(&self) -> Self::Node {
        START_BLOCK
    }
}

impl<'tcx> graph::WithSuccessors for BasicBlocks<'tcx> {
    #[inline]
    fn successors(&self, node: Self::Node) -> <Self as graph::GraphSuccessors<'_>>::Iter {
        self.basic_blocks[node].terminator().successors()
    }
}

impl<'a, 'b> graph::GraphSuccessors<'b> for BasicBlocks<'a> {
    type Item = BasicBlock;
    type Iter = Successors<'b>;
}

impl<'tcx, 'graph> graph::GraphPredecessors<'graph> for BasicBlocks<'tcx> {
    type Item = BasicBlock;
    type Iter = std::iter::Copied<std::slice::Iter<'graph, BasicBlock>>;
}

impl<'tcx> graph::WithPredecessors for BasicBlocks<'tcx> {
    #[inline]
    fn predecessors(&self, node: Self::Node) -> <Self as graph::GraphPredecessors<'_>>::Iter {
        self.predecessors()[node].iter().copied()
    }
}

TrivialTypeTraversalAndLiftImpls! {
    Cache,
}

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

impl<CTX> HashStable<CTX> for Cache {
    #[inline]
    fn hash_stable(&self, _: &mut CTX, _: &mut StableHasher) {}
}
