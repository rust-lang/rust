use crate::mir::graph_cyclic_cache::GraphIsCyclicCache;
use crate::mir::predecessors::{PredecessorCache, Predecessors};
use crate::mir::switch_sources::{SwitchSourceCache, SwitchSources};
use crate::mir::traversal::PostorderCache;
use crate::mir::{BasicBlock, BasicBlockData, Successors, START_BLOCK};

use rustc_data_structures::graph;
use rustc_data_structures::graph::dominators::{dominators, Dominators};
use rustc_index::vec::IndexVec;

#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable, TypeFoldable, TypeVisitable)]
pub struct BasicBlocks<'tcx> {
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    predecessor_cache: PredecessorCache,
    switch_source_cache: SwitchSourceCache,
    is_cyclic: GraphIsCyclicCache,
    postorder_cache: PostorderCache,
}

impl<'tcx> BasicBlocks<'tcx> {
    #[inline]
    pub fn new(basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>) -> Self {
        BasicBlocks {
            basic_blocks,
            predecessor_cache: PredecessorCache::new(),
            switch_source_cache: SwitchSourceCache::new(),
            is_cyclic: GraphIsCyclicCache::new(),
            postorder_cache: PostorderCache::new(),
        }
    }

    /// Returns true if control-flow graph contains a cycle reachable from the `START_BLOCK`.
    #[inline]
    pub fn is_cfg_cyclic(&self) -> bool {
        self.is_cyclic.is_cyclic(self)
    }

    #[inline]
    pub fn dominators(&self) -> Dominators<BasicBlock> {
        dominators(&self)
    }

    /// Returns predecessors for each basic block.
    #[inline]
    pub fn predecessors(&self) -> &Predecessors {
        self.predecessor_cache.compute(&self.basic_blocks)
    }

    /// Returns basic blocks in a postorder.
    #[inline]
    pub fn postorder(&self) -> &[BasicBlock] {
        self.postorder_cache.compute(&self.basic_blocks)
    }

    /// `switch_sources()[&(target, switch)]` returns a list of switch
    /// values that lead to a `target` block from a `switch` block.
    #[inline]
    pub fn switch_sources(&self) -> &SwitchSources {
        self.switch_source_cache.compute(&self.basic_blocks)
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
        self.predecessor_cache.invalidate();
        self.switch_source_cache.invalidate();
        self.is_cyclic.invalidate();
        self.postorder_cache.invalidate();
    }
}

impl<'tcx> std::ops::Deref for BasicBlocks<'tcx> {
    type Target = IndexVec<BasicBlock, BasicBlockData<'tcx>>;

    #[inline]
    fn deref(&self) -> &IndexVec<BasicBlock, BasicBlockData<'tcx>> {
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
