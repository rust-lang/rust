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
    pub biggest_scc: usize,
    /// The SCC of each block.
    pub components: IndexVec<BasicBlock, u32>,
    // pub components: Vec<usize>,
    // pub components: Vec<isize>,
    /// The contents of each SCC: its blocks, in RPO.
    pub sccs: Vec<SmallVec<[BasicBlock; 2]>>,
    // pub sccs: Vec<Vec<BasicBlock>>,
    // pub queue: Vec<usize>,
    pub queue: Vec<u32>,
}

// impl SccData {
//     /// Returns the SCC index for a given `block`.
//     #[inline(always)]
//     pub fn scc(&self, block: BasicBlock) -> u32 {
//         self.components[block]
//     }

//     /// An iterator of the blocks that belong to the given SCC. Crucially, the blocks within the SCC
//     /// are ordered in RPO.
//     #[inline(always)]
//     pub fn blocks_in_rpo(&self, scc: usize) -> impl Iterator<Item = BasicBlock> {
//         self.sccs[scc].iter().copied()
//     }
// }

use std::collections::VecDeque;

use rustc_index::Idx;
use rustc_index::bit_set::DenseBitSet;

// struct PearceRecursive {
//     r_index: Vec<usize>,
//     stack: VecDeque<usize>,
//     index: usize,
//     c: usize,
// }

// impl PearceRecursive {
//     fn new(node_count: usize) -> Self {
//         assert!(node_count > 0); // only a non-empty graph is supported
//         // todo: assert node_count is within overflow limits
//         Self { r_index: vec![0; node_count], stack: VecDeque::new(), index: 1, c: node_count - 1 }
//     }

//     fn compute_sccs(&mut self, blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
//         for (block_idx, _) in blocks.iter_enumerated() {
//             let v = block_idx.as_usize();
//             if self.r_index[v] == 0 {
//                 // v is unvisited
//                 self.visit(v, blocks);
//             }
//         }

//         // The SCC labels are from N - 1 to zero, remap them from 0 to the component count, to match
//         // their position in an array of SCCs.
//         let node_count = blocks.len() - 1;
//         for scc_index in self.r_index.iter_mut() {
//             *scc_index = node_count - *scc_index;
//         }
//     }

//     fn visit(&mut self, v: usize, blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
//         let mut root = true;
//         self.r_index[v] = self.index;
//         self.index += 1; // todo: overflow

//         for succ in blocks[BasicBlock::from_usize(v)].terminator().successors() {
//             let w = succ.as_usize();
//             if self.r_index[w] == 0 {
//                 self.visit(w, blocks);
//             }
//             if self.r_index[w] < self.r_index[v] {
//                 self.r_index[v] = self.r_index[w];
//                 root = false;
//             }
//         }

//         if root {
//             self.index -= 1; // todo: underflow

//             while let Some(&w) = self.stack.front()
//                 && self.r_index[v] <= self.r_index[w]
//             {
//                 self.stack.pop_front();
//                 self.r_index[w] = self.c;
//                 self.index -= 1; // todo: underflow
//             }

//             self.r_index[v] = self.c;
//             self.c -= 1; // todo: underflow
//         } else {
//             self.stack.push_front(v);
//         }
//     }
// }

struct PearceRecursive {
    r_index: IndexVec<BasicBlock, u32>,
    stack: VecDeque<BasicBlock>,
    index: u32,
    c: u32,
    // duplicates_count: usize,
    // in_stack: FxHashSet<BasicBlock>,
    // youpi: Vec<SmallVec<[BasicBlock; 2]>>,
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

            // duplicates_count: 0,
            // in_stack: FxHashSet::default(),
            // youpi: Vec::new(),
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

            // let c = self.r_index.len() - self.c;
            // let mut contents = SmallVec::new();
            // contents.push(v);

            while let Some(&w) = self.stack.front()
                && self.r_index[v] <= self.r_index[w]
            {
                self.stack.pop_front();
                // self.in_stack.remove(&w);
                self.r_index[w] = self.c;
                self.index -= 1;
                // contents.push(w); // maybe should also sort contents by r_index?
            }

            self.r_index[v] = self.c;

            // self.youpi.push(contents);
        } else {
            // // FIXME: check if tracking what's already inside the stack to avoid duplicate work is
            // // worth it.
            // if self.stack.contains(&v) {
            //     self.duplicates_count += 1;
            // }

            // Note: tracking what's in the stack wasn't worth it on cranelift-codegen (there are no
            // dupes), maybe for other benchmarks?
            // FIXME: try
            // if self.in_stack.insert(v) {
            //     self.stack.push_front(v);
            // }

            self.stack.push_front(v);
        }
    }
}

// struct PearceIterative {
//     r_index: Vec<usize>,
//     index: usize,
//     c: usize,
//     stack_vertex: VecDeque<usize>,
//     stack_iteration: VecDeque<usize>,
//     root: Vec<bool>,
//     // root: DenseBitSet<usize>,
// }

// impl PearceIterative {
//     fn new(node_count: usize) -> Self {
//         // assert!(node_count > 0); // only a non-empty graph is supported
//         // todo: assert node_count is within overflow limits
//         Self {
//             r_index: vec![0; node_count],
//             // stack_vertex: VecDeque::new(),
//             // stack_iteration: VecDeque::new(),
//             stack_vertex: VecDeque::with_capacity(node_count),
//             stack_iteration: VecDeque::with_capacity(node_count),
//             index: 1,
//             c: node_count - 1,
//             root: vec![false; node_count],
//             // root: DenseBitSet::new_empty(node_count),
//         }
//     }

//     fn compute_sccs(&mut self, blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
//         for (block_idx, _) in blocks.iter_enumerated() {
//             let v = block_idx.as_usize();
//             if self.r_index[v] == 0 {
//                 self.visit(v, blocks);
//             }
//         }

//         // The computed SCC labels are from N - 1 to zero, remap them from 0 to the component count,
//         // to match their position in an array of SCCs.
//         let node_count = blocks.len() - 1;
//         for scc_index in self.r_index.iter_mut() {
//             *scc_index = node_count - *scc_index;
//         }

//         // Adjust the component index counter to the component count
//         self.c = node_count - self.c;
//     }

//     fn visit(&mut self, v: usize, blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
//         // procedure visit(v)
//         //     6: beginVisiting(v)
//         //     7: while vS != ∅ do
//         //     8: visitLoop()
//         self.begin_visiting(v, blocks);
//         while !self.stack_vertex.is_empty() {
//             self.visit_loop(blocks);
//         }
//     }

//     // inline this
//     fn visit_loop(&mut self, blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
//         // procedure visitLoop()
//         //     9: v = top(vS) ; i = top(iS)
//         //    10: while i ≤ |E(v)| do
//         //    11:   if i > 0 then finishEdge(v, i − 1)
//         //    12:   if i < |E(v)| ∧ beginEdge(v, i) then return
//         //    13:   i = i + 1
//         //    14: finishVisiting(v)
//         let v = *self.stack_vertex.front().unwrap();
//         let mut i = *self.stack_iteration.front().unwrap();

//         // todo: match on edges() to get the count
//         let v_successor_count = blocks[BasicBlock::from_usize(v)].terminator().successors().count();
//         // assert!(i <= v_successor_count, "i: {i}, v_successor_count: {v_successor_count}");

//         while i <= v_successor_count {
//             if i > 0 {
//                 self.finish_edge(v, i - 1, blocks);
//             }
//             if i < v_successor_count && self.begin_edge(v, i, blocks) {
//                 return;
//             }
//             i += 1;
//         }
//         self.finish_visiting(v, blocks);
//     }

//     fn begin_visiting(&mut self, v: usize, _blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
//         // procedure beginVisiting(v)
//         //     15: push(vS, v) ; push(iS, 0)
//         //     16: root[v] = true ; rindex[v] = index ; index = index + 1
//         self.stack_vertex.push_front(v);
//         self.stack_iteration.push_front(0);
//         self.root[v] = true;
//         // self.root.insert(v);
//         self.r_index[v] = self.index;
//         self.index += 1;
//     }

//     // inline this
//     fn finish_visiting(&mut self, v: usize, _blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
//         // procedure finishVisiting(v)
//         //    17: pop(vS) ; pop(iS)
//         //    18: if root[v] then
//         //    19:   index = index − 1
//         //    20:   while vS != ∅ ∧ rindex[v] ≤ rindex[top(vS)] do
//         //    21:       w = pop(vS)
//         //    22:       rindex[w] = c
//         //    23:       index = index − 1
//         //    24:   rindex[v] = c
//         //    25:   c = c − 1
//         //    26: else
//         //    27:   push(vS, v)
//         self.stack_vertex.pop_front();
//         self.stack_iteration.pop_front();
//         if self.root[v] {
//             // if self.root.contains(v) {
//             self.index -= 1;
//             while let Some(&w) = self.stack_vertex.back()
//                 && self.r_index[v] <= self.r_index[w]
//             {
//                 self.stack_vertex.pop_back();
//                 self.r_index[w] = self.c;
//                 self.index -= 1;
//             }
//             self.r_index[v] = self.c;
//             self.c -= 1;
//         } else {
//             self.stack_vertex.push_back(v);
//         }
//     }

//     // finish this
//     fn begin_edge(
//         &mut self,
//         v: usize,
//         k: usize,
//         blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>,
//     ) -> bool {
//         // procedure beginEdge(v, k)
//         //    28: w = E(v)[k]
//         //    29: if rindex[w] == 0 then
//         //    30:   pop(iS) ; push(iS, k + 1)
//         //    31:   beginVisiting(w)
//         //    32:   return true
//         //    33: else
//         //    34:   return false

//         let w = blocks[BasicBlock::from_usize(v)].terminator().successors().nth(k).unwrap();
//         let w = w.as_usize();
//         if self.r_index[w] == 0 {
//             self.stack_iteration.pop_front();
//             self.stack_iteration.push_front(k + 1);
//             self.begin_visiting(w, blocks);
//             return true;
//         } else {
//             return false;
//         }
//     }

//     // inline this
//     fn finish_edge(
//         &mut self,
//         v: usize,
//         k: usize,
//         blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>,
//     ) {
//         // procedure finishEdge(v, k)
//         //     35: w = E(v)[k]
//         //     36: if rindex[w] < rindex[v] then rindex[v] = rindex[w] ; root[v] = false
//         let w = blocks[BasicBlock::from_usize(v)].terminator().successors().nth(k).unwrap();
//         let w = w.as_usize();
//         if self.r_index[w] < self.r_index[v] {
//             self.r_index[v] = self.r_index[w];
//             self.root[v] = false;
//             // self.root.remove(v);
//         }
//     }
// }

// struct Scc {
//     candidate_component_roots: Vec<usize>,
//     components: Vec<isize>,
//     component_count: usize,
//     dfs_numbers: Vec<u32>,
//     d: u32,
//     stack: VecDeque<usize>,
//     visited: DenseBitSet<usize>,
//     // queue: Vec<usize>, // can compute this while discovering components?!,?!
// }

// impl Scc {
//     fn new(node_count: usize) -> Self {
//         Self {
//             candidate_component_roots: vec![0; node_count],
//             components: vec![-1; node_count],
//             component_count: 0,
//             dfs_numbers: vec![0; node_count],
//             d: 0,
//             stack: VecDeque::new(),
//             visited: DenseBitSet::new_empty(node_count),
//             // queue: Vec::new(),
//         }
//     }

//     fn compute_sccs(&mut self, blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
//         for (idx, block) in blocks.iter_enumerated() {
//             let edges = block.terminator().edges();
//             if matches!(edges, rustc_middle::mir::TerminatorEdges::None) {
//                 continue;
//             }

//             let idx = idx.as_usize();
//             if !self.visited.contains(idx) {
//                 self.dfs_visit(idx, blocks);
//             }
//         }
//     }

//     fn dfs_visit(&mut self, v: usize, blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>) {
//         self.candidate_component_roots[v] = v;
//         self.components[v] = -1;

//         self.d += 1;
//         self.dfs_numbers[v] = self.d;

//         self.visited.insert(v);

//         let idx = BasicBlock::from_usize(v);
//         for succ in blocks[idx].terminator().successors() {
//             let w = succ.as_usize();

//             if !self.visited.contains(w) {
//                 self.dfs_visit(w, blocks);
//             }

//             if self.components[w] == -1
//                 && self.dfs_numbers[self.candidate_component_roots[w]]
//                     < self.dfs_numbers[self.candidate_component_roots[v]]
//             {
//                 self.candidate_component_roots[v] = self.candidate_component_roots[w];
//             }
//         }

//         if self.candidate_component_roots[v] == v {
//             self.components[v] = self.component_count as isize;
//             self.component_count += 1;

//             // We have discovered a component.
//             // self.queue.push(self.component_count);

//             while self.stack.front().is_some()
//                 && self.dfs_numbers[*self.stack.front().expect("peek front failed")]
//                     > self.dfs_numbers[v]
//             {
//                 let w = self.stack.pop_front().expect("pop front failed");
//                 self.components[w] = self.components[v];
//             }
//         } else {
//             self.stack.push_front(v);
//         }
//     }
// }

#[derive(Clone, Debug)]
struct VecQueue<T: Idx> {
    queue: Vec<T>,
    set: DenseBitSet<T>,
}

// impl<T: Idx> Default for VecQueue<T> {
//     fn default() -> Self {
//         Self { queue: Default::default(), set: DenseBitSet::new_empty(0) }
//     }
// }

impl<T: Idx> VecQueue<T> {
    #[inline]
    fn with_none(len: usize) -> Self {
        VecQueue { queue: Vec::with_capacity(len), set: DenseBitSet::new_empty(len) }
    }

    #[inline]
    fn insert(&mut self, element: T) {
        if self.set.insert(element) {
            self.queue.push(element);
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

    // #[inline]
    // pub fn sccs(&self) -> &SccData {
    //     self.cache.sccs.get_or_init(|| {
    //         let block_count = self.basic_blocks.len();

    //         let mut pearce = PearceRecursive::new(block_count);
    //         pearce.compute_sccs(&self.basic_blocks);
    //         let component_count = pearce.c as usize;

    //         let mut sccs = vec![smallvec::SmallVec::new(); component_count];
    //         for &block in self.reverse_postorder().iter() {
    //             let scc = pearce.r_index[block] as usize;
    //             sccs[scc].push(block);
    //         }
    //         SccData { component_count, components: pearce.r_index, sccs }
    //     })
    // }

    #[inline]
    pub fn sccs(&self) -> &SccData {
        self.cache.sccs.get_or_init(|| {
            // let mut sccs = Scc::new(self.basic_blocks.len());
            // sccs.compute_sccs(&self.basic_blocks);

            // let component_count = sccs.component_count;

            // let mut components=
            //     vec![smallvec::SmallVec::<[BasicBlock; 2]>::new(); component_count];
            // // let mut components = vec![Vec::new(); component_count];
            // let mut scc_queue = VecQueue::with_none(component_count);

            // // Reuse block bitset as scc bitset
            // // sccs.visited.clear();

            // for &block in self.reverse_postorder().iter() {
            //     let scc = sccs.components[block.as_usize()] as usize;
            //     components[scc].push(block);

            //     // if sccs.visited.insert(scc) {
            //     //     scc_queue.push(scc);
            //     // }
            //     scc_queue.insert(scc);
            // }

            // SccData {
            //     components: sccs.components,
            //     component_count,
            //     sccs: components,
            //     queue: scc_queue.queue,
            // }

            // // ---

            // let mut sccs = Scc::new(self.basic_blocks.len());
            // sccs.compute_sccs(&self.basic_blocks);

            // let component_count = sccs.component_count;

            // let mut components = vec![Vec::new(); component_count];
            // for &block in self.reverse_postorder().iter() {
            //     let scc = sccs.components[block.as_usize()] as usize;
            //     components[scc].push(block);
            // }

            // // let mut scc_queue = VecQueue::with_none(component_count);
            // // for scc in
            // //     self.reverse_postorder().iter().map(|bb| sccs.components[bb.as_usize()] as usize)
            // // {
            // //     scc_queue.insert(scc);
            // // }

            // // SccData { components: sccs.components, sccs: components, queue: scc_queue.queue }
            // SccData { component_count, components: sccs.components, sccs: components }

            // ---
            // use rustc_data_structures::graph::scc::*;

            // struct Wrapper<'a, 'tcx> {
            //     x: &'a IndexVec<BasicBlock, BasicBlockData<'tcx>>,
            // }
            // impl graph::DirectedGraph for Wrapper<'_, '_> {
            //     type Node = BasicBlock;

            //     #[inline]
            //     fn num_nodes(&self) -> usize {
            //         self.x.len()
            //     }
            // }
            // impl graph::Successors for Wrapper<'_, '_> {
            //     #[inline]
            //     fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
            //         self.x[node].terminator().successors()
            //     }
            // }
            // let wrapper = Wrapper { x: &self.basic_blocks };
            // let rustc_sccs = Sccs::<BasicBlock, u32>::new(&wrapper);

            // let component_count = rustc_sccs.num_sccs();

            // // let mut sccs = vec![Vec::new(); component_count];
            // let mut sccs = vec![smallvec::SmallVec::new(); component_count];
            // for &block in self.reverse_postorder().iter() {
            //     let scc = rustc_sccs.scc(block) as usize;
            //     sccs[scc].push(block);
            // }

            // let mut scc_queue = VecQueue::with_none(component_count);
            // for scc in
            //     // self.reverse_postorder().iter().map(|&bb| pearce.r_index[bb])
            //     self.reverse_postorder().iter().map(|&bb| rustc_sccs.scc(bb) as u32)
            // {
            //     scc_queue.insert(scc);
            // }

            // return SccData {
            //     component_count,
            //     sccs,
            //     components: rustc_sccs.scc_indices,
            //     biggest_scc: 0,
            //     queue: scc_queue.queue,
            // };

            // ---

            // let mut edges = Vec::new();

            // for (bb, block) in self.basic_blocks.iter_enumerated() {
            //     for succ in block.terminator().successors() {
            //         edges.push((bb.as_usize(), succ.as_usize()));
            //     }
            // }

            // let graph: petgraph::Graph<usize, usize, petgraph::Directed, usize> = petgraph::Graph::from_edges(edges);
            // let sccz = petgraph::algo::tarjan_scc(&graph);
            // let component_count = sccz.len();

            // let mut components = vec![0_isize; self.basic_blocks.len()];
            // for (scc_idx, scc_contents) in sccz.iter().enumerate() {
            //     for block_idx in scc_contents {
            //         components[block_idx.index()] = scc_idx as isize;
            //     }
            // }

            // let mut sccs = vec![Vec::new(); component_count];
            // for &block in self.reverse_postorder().iter() {
            //     let scc = components[block.as_usize()] as usize;
            //     sccs[scc].push(block);
            // }

            // SccData { component_count, sccs, components }

            // ---
            let block_count = self.basic_blocks.len();

            // tarjan is incorrect on some of the tests/ui/async-await/async-drop/ tests
            // let mut tarjan = Scc::new(block_count);
            // tarjan.compute_sccs(&self.basic_blocks);

            let mut pearce = PearceRecursive::new(block_count);
            pearce.compute_sccs(&self.basic_blocks);
            let component_count = pearce.c as usize;
            // eprintln!("pearce recursive duplicate count: {}", pearce.duplicates_count);

            // let mut pearce_i = PearceIterative::new(block_count);
            // pearce_i.compute_sccs(&self.basic_blocks);
            // let component_count = pearce_i.c as usize;

            // // // assert_eq!(rustc_sccs.num_sccs(), tarjan.component_count);
            // assert_eq!(rustc_sccs.num_sccs(), pearce.c as usize);
            // assert_eq!(rustc_sccs.num_sccs(), pearce_i.c);

            // for block in self.basic_blocks.indices() {
            //     assert_eq!(
            //         rustc_sccs.scc(block),
            //         tarjan.components[block.as_usize()] as usize
            //     );
            // }

            // assert_eq!(rustc_sccs.num_sccs(), pearce.c as usize);
            // for block in self.basic_blocks.indices() {
            //     assert_eq!(rustc_sccs.scc(block), pearce.r_index[block] as usize);
            // }

            // assert_eq!(rustc_sccs.num_sccs(), pearce_i.c);
            // for block in self.basic_blocks.indices() {
            //     assert_eq!(rustc_sccs.scc(block), pearce_i.r_index[block.as_usize()] as usize);
            // }

            // for &block in self.reverse_postorder().iter() {
            //     assert_eq!(rustc_sccs.scc(block), pearce.r_index[block] as usize);
            //     assert_eq!(rustc_sccs.scc(block), pearce_i.r_index[block.as_usize()] as usize);
            //     assert!(rustc_sccs.scc(block) < rustc_sccs.num_sccs());
            //     assert!(pearce.r_index[block] < pearce.c);
            //     assert!(pearce_i.r_index[block.as_usize()] < pearce_i.c);
            // }

            // assert_eq!(
            //     tarjan.components,
            //     pearce.r_index.iter().map(|&c| c as isize).collect::<Vec<_>>(),
            //     "blocks: {}, scc components: {}; pearce index: {}, c: {}",
            //     block_count,
            //     tarjan.component_count,
            //     pearce.index,
            //     pearce.c,
            // );

            // assert_eq!(
            //     pearce.r_index,
            //     pearce_i.r_index,
            //     "blocks: {}, scc components: {}; pearce_r index: {}, c: {}; pearce_i index: {}, c: {}",
            //     block_count,
            //     tarjan.component_count,
            //     pearce.index,
            //     pearce.c,
            //     pearce_i.index,
            //     pearce_i.c,
            // );

            // let component_count = tarjan.component_count;

            // let mut sccs = vec![Vec::new(); component_count];
            // for &block in self.reverse_postorder().iter() {
            //     let scc = tarjan.components[block.as_usize()] as usize;
            //     sccs[scc].push(block);
            // }

            // let component_count = block_count - pearce.c - 1;
            // let mut sccs = vec![Vec::new(); component_count];
            // for &block in self.reverse_postorder().iter() {
            //     let scc = pearce.r_index[block.as_usize()] as usize;
            //     sccs[scc].push(block);
            // }

            // for block in self.basic_blocks.indices() {
            //     assert_eq!(rustc_sccs.scc(block), pearce.r_index[block] as usize);
            // }

            // assert_eq!(
            //     self.basic_blocks
            //         .indices()
            //         .map(|bb| rustc_sccs.scc(bb) as u32)
            //         .collect::<IndexVec<_, _>>(),
            //     pearce.r_index,
            //     "blocks: {}, scc components: {}; pearce index: {}, c: {}",
            //     block_count,
            //     rustc_sccs.num_sccs(),
            //     pearce.index,
            //     pearce.c,
            // );
            // assert_eq!(
            //     self.basic_blocks
            //         .indices()
            //         .map(|bb| rustc_sccs.scc(bb) as u32)
            //         .collect::<IndexVec<_, _>>(),
            //     pearce.r_index,
            //     "blocks: {}, scc components: {}; pearce_i index: {}, c: {}",
            //     block_count,
            //     rustc_sccs.num_sccs(),
            //     pearce_i.index,
            //     pearce_i.c,
            // );

            // assert_eq!(
            //     rustc_sccs.num_sccs(),
            //     component_count,
            //     "blocks: {}, scc components: {}; pearce index: {}, c: {}",
            //     block_count,
            //     rustc_sccs.num_sccs(),
            //     pearce.index,
            //     pearce.c,
            // );
            // assert!(component_count > 0);
            // assert!(
            //     component_count <= block_count,
            //     "component count: {}, block count: {}, pearse c: {}",
            //     component_count,
            //     block_count,
            //     pearce.c
            // );

            // let mut sccs = vec![Vec::new(); component_count];
            let mut sccs = vec![smallvec::SmallVec::new(); component_count];
            for &block in self.reverse_postorder().iter() {
                let scc = pearce.r_index[block] as usize;
                // let scc = pearce_i.r_index[block.as_usize()] as usize;
                sccs[scc].push(block);
            }

            // if self.basic_blocks.len() < 200 {
            //     for scc in 0..component_count {
            //         if pearce.youpi[scc][..] != sccs[scc][..] {
            //             eprintln!(
            //                 "scc: {scc}, pearce: {:?}, post: {:?}",
            //                 pearce.youpi[scc], sccs[scc]
            //             );
            //         }
            //     }
            // }

            let biggest_scc = 0; //sccs.iter().map(|scc| scc.len()).max().unwrap();

            let mut scc_queue = VecQueue::with_none(component_count);
            for scc in self.reverse_postorder().iter().map(|&bb| pearce.r_index[bb])
            // self.reverse_postorder().iter().map(|&bb| pearce_i.r_index[bb.as_usize()] as u32)
            {
                scc_queue.insert(scc);
            }

            // SccData { component_count, components: tarjan.components, sccs }
            SccData {
                component_count,
                components: pearce.r_index,
                // sccs: pearce.youpi, // 1351402 invalidations -- FIXME: which is fastest? FIXME: try sorting within youpi vecs, per r_index?
                sccs, // 1327320 invalidations
                biggest_scc,
                queue: scc_queue.queue,
            }
            // SccData { component_count, components: pearce_i.r_index, sccs, biggest_scc, queue: scc_queue.queue }
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
