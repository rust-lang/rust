use std::sync::OnceLock;

use rustc_data_structures::graph;
use rustc_data_structures::graph::dominators::{Dominators, dominators};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_index::{IndexSlice, IndexVec};
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use smallvec::SmallVec;

use crate::mir::traversal::Postorder;
use crate::mir::{BasicBlock, BasicBlockData, START_BLOCK};

#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable, TypeFoldable, TypeVisitable)]
pub struct BasicBlocks<'tcx> {
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    cache: Cache,
}

// Typically 95%+ of basic blocks have 4 or fewer predecessors.
type Predecessors = IndexVec<BasicBlock, SmallVec<[BasicBlock; 4]>>;

#[derive(Debug, Clone, Copy)]
pub enum SwitchTargetValue {
    // A normal switch value.
    Normal(u128),
    // The final "otherwise" fallback value.
    Otherwise,
}

#[derive(Clone, Default, Debug)]
struct Cache {
    predecessors: OnceLock<Predecessors>,
    reverse_postorder: OnceLock<Vec<BasicBlock>>,
    dominators: OnceLock<Dominators<BasicBlock>>,
    is_cyclic: OnceLock<bool>,
    sccs: OnceLock<SccData>,
}

#[derive(Clone, Default, Debug)]
pub struct SccData {
    pub component_count: usize,

    /// The SCC of each block.
    pub components: IndexVec<BasicBlock, u32>,

    /// The contents of each SCC: its blocks, in RPO.
    pub sccs: Vec<SmallVec<[BasicBlock; 2]>>,
}

use std::collections::VecDeque;

struct PearceRecursive {
    r_index: IndexVec<BasicBlock, u32>,
    stack: VecDeque<BasicBlock>,
    index: u32,
    c: u32,
}

impl PearceRecursive {
    fn new(node_count: usize) -> Self {
        assert!(node_count > 0); // only a non-empty graph is supported
        // todo: assert node_count is within overflow limits
        Self {
            r_index: IndexVec::from_elem_n(0, node_count),
            stack: VecDeque::new(),
            index: 1,
            c: node_count.try_into().unwrap(),
            // c: node_count - 1,
        }
    }

    fn compute_sccs(&mut self, blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
        for v in blocks.indices() {
            if self.r_index[v] == 0 {
                self.visit(v, blocks);
            }
        }

        // The SCC labels are from N - 1 to zero, remap them from 0 to the component count, to match
        // their position in an array of SCCs.
        let node_count: u32 = blocks.len().try_into().unwrap();
        for scc_index in self.r_index.iter_mut() {
            *scc_index = node_count - *scc_index - 1;
        }

        // Adjust the component index counter to the component count
        self.c = node_count - self.c;
    }

    fn visit(&mut self, v: BasicBlock, blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
        let mut root = true;
        self.r_index[v] = self.index;
        self.index += 1;

        for w in blocks[v].terminator().successors() {
            if self.r_index[w] == 0 {
                self.visit(w, blocks);
            }
            if self.r_index[w] < self.r_index[v] {
                self.r_index[v] = self.r_index[w];
                root = false;
            }
        }

        if root {
            self.index -= 1;
            self.c -= 1;

            while let Some(&w) = self.stack.front()
                && self.r_index[v] <= self.r_index[w]
            {
                self.stack.pop_front();
                self.r_index[w] = self.c;
                self.index -= 1;
            }

            self.r_index[v] = self.c;
        } else {
            self.stack.push_front(v);
        }
    }
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

    #[inline]
    pub fn dominators(&self) -> &Dominators<BasicBlock> {
        self.cache.dominators.get_or_init(|| dominators(self))
    }

    #[inline]
    pub fn sccs(&self) -> &SccData {
        self.cache.sccs.get_or_init(|| {
            let block_count = self.basic_blocks.len();

            let mut pearce = PearceRecursive::new(block_count);
            pearce.compute_sccs(&self.basic_blocks);
            let component_count = pearce.c as usize;

            let mut sccs = vec![smallvec::SmallVec::new(); component_count];
            for &block in self.reverse_postorder().iter() {
                let scc = pearce.r_index[block] as usize;
                sccs[scc].push(block);
            }
            SccData { component_count, components: pearce.r_index, sccs }
        })
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
    /// See [`traversal::reverse_postorder`]'s docs to learn what is preorder traversal.
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
TrivialTypeTraversalImpls! { Cache }

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
