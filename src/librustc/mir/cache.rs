use rustc_index::vec::IndexVec;
//use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
//use rustc_serialize::{Encodable, Encoder, Decodable, Decoder};
//use crate::ich::StableHashingContext;
use crate::mir::{BasicBlock, BasicBlockData, Body, LocalDecls, Location, Successors};
use rustc_data_structures::graph::{self, GraphPredecessors, GraphSuccessors};
use rustc_data_structures::graph::dominators::{dominators, Dominators};
use std::iter;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::vec::IntoIter;

#[derive(Clone, Debug)]
pub struct Cache {
    predecessors: Option<IndexVec<BasicBlock, Vec<BasicBlock>>>,
}


//impl<'tcx, T> rustc_serialize::Encodable for Cache<'tcx, T> {
//    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
//        Encodable::encode(&(), s)
//    }
//}
//
//impl<'tcx, T> rustc_serialize::Decodable for Cache<'tcx, T> {
//    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
//        Decodable::decode(d).map(|_v: ()| Self::new())
//    }
//}
//
//impl<'a, 'tcx, T> HashStable<StableHashingContext<'a>> for Cache<'tcx, T> {
//    fn hash_stable(&self, _: &mut StableHashingContext<'a>, _: &mut StableHasher) {
//        // Do nothing.
//    }
//}

impl Cache {
    pub fn new() -> Self {
        Self {
            predecessors: None,
        }
    }

    #[inline]
    pub fn invalidate_predecessors(&mut self) {
        // FIXME: consider being more fine-grained
        self.predecessors = None;
    }

    #[inline]
    /// This will recompute the predecessors cache if it is not available
    pub fn predecessors(&mut self, body: &Body<'_>) -> &IndexVec<BasicBlock, Vec<BasicBlock>> {
        if self.predecessors.is_none() {
            let mut result = IndexVec::from_elem(vec![], body.basic_blocks());
            for (bb, data) in body.basic_blocks().iter_enumerated() {
                if let Some(ref term) = data.terminator {
                    for &tgt in term.successors() {
                        result[tgt].push(bb);
                    }
                }
            }

            self.predecessors = Some(result)
        }

        self.predecessors.as_ref().unwrap()
    }

    #[inline]
    pub fn predecessors_for(&mut self, bb: BasicBlock, body: &Body<'_>) -> &[BasicBlock] {
        &self.predecessors(body)[bb]
    }

    #[inline]
    pub fn predecessor_locations<'a>(&'a mut self, loc: Location, body: &'a Body<'a>) -> impl Iterator<Item = Location> + 'a {
        let if_zero_locations = if loc.statement_index == 0 {
            let predecessor_blocks = self.predecessors_for(loc.block, body);
            let num_predecessor_blocks = predecessor_blocks.len();
            Some(
                (0..num_predecessor_blocks)
                    .map(move |i| predecessor_blocks[i])
                    .map(move |bb| body.terminator_loc(bb)),
            )
        } else {
            None
        };

        let if_not_zero_locations = if loc.statement_index == 0 {
            None
        } else {
            Some(Location { block: loc.block, statement_index: loc.statement_index - 1 })
        };

        if_zero_locations.into_iter().flatten().chain(if_not_zero_locations)
    }

    #[inline]
    pub fn basic_blocks_mut<'a, 'tcx>(&mut self, body: &'a mut Body<'tcx>) -> &'a mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        debug!("bbm: Clearing predecessors cache for body at: {:?}", body.span.data());
        self.invalidate_predecessors();
        &mut body.basic_blocks
    }

    #[inline]
    pub fn basic_blocks_and_local_decls_mut<'a, 'tcx>(
        &mut self,
        body: &'a mut Body<'tcx>
    ) -> (&'a mut IndexVec<BasicBlock, BasicBlockData<'tcx>>, &'a mut LocalDecls<'tcx>) {
        debug!("bbaldm: Clearing predecessors cache for body at: {:?}", body.span.data());
        self.invalidate_predecessors();
        (&mut body.basic_blocks, &mut body.local_decls)
    }
}

pub struct BodyCache<T> {
    cache: Cache,
    body: T,
}

impl<T> BodyCache<T> {
    pub fn new(body: T) -> Self {
        Self {
            cache: Cache::new(),
            body
        }
    }
}

impl<'a, 'tcx> BodyCache<&'a Body<'tcx>> {
    #[inline]
    pub fn predecessors_for(&mut self, bb: BasicBlock) -> &[BasicBlock] {
        self.cache.predecessors_for(bb, self.body)
    }

    #[inline]
    pub fn body(&self) -> &'a Body<'tcx> {
        self.body
    }

    #[inline]
    pub fn basic_blocks(&self) -> &IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        &self.body.basic_blocks
    }

    #[inline]
    pub fn dominators(&mut self) -> Dominators<BasicBlock> {
        dominators(self)
    }
}

impl<'a, 'tcx> Deref for BodyCache<&'a Body<'tcx>> {
    type Target = Body<'tcx>;

    fn deref(&self) -> &Self::Target {
        self.body
    }
}

impl<'a, 'tcx> Index<BasicBlock> for BodyCache<&'a Body<'tcx>> {
    type Output = BasicBlockData<'tcx>;

    #[inline]
    fn index(&self, index: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.body[index]
    }
}

impl<'a, 'tcx> graph::DirectedGraph for BodyCache<&'a Body<'tcx>> {
    type Node = BasicBlock;
}

impl<'a, 'graph, 'tcx> graph::GraphPredecessors<'graph> for BodyCache<&'a Body<'tcx>> {
    type Item = BasicBlock;
    type Iter = IntoIter<BasicBlock>;
}

impl<'a, 'tcx> graph::WithPredecessors for BodyCache<&'a Body<'tcx>> {
    fn predecessors(
        &mut self,
        node: Self::Node,
    ) -> <Self as GraphPredecessors<'_>>::Iter {
        self.predecessors_for(node).to_vec().into_iter()
    }
}

impl<'a, 'tcx> graph::WithNumNodes for BodyCache<&'a Body<'tcx>> {
    fn num_nodes(&self) -> usize {
        self.body.num_nodes()
    }
}

impl<'a, 'tcx> graph::WithStartNode for BodyCache<&'a Body<'tcx>> {
    fn start_node(&self) -> Self::Node {
        self.body.start_node()
    }
}

impl<'a, 'tcx> graph::WithSuccessors for BodyCache<&'a Body<'tcx>> {
    fn successors(
        &self,
        node: Self::Node,
    ) -> <Self as GraphSuccessors<'_>>::Iter {
        self.body.successors(node)
    }
}

impl<'a, 'b, 'tcx> graph::GraphSuccessors<'b> for BodyCache<&'a Body<'tcx>> {
    type Item = BasicBlock;
    type Iter = iter::Cloned<Successors<'b>>;
}

impl<'a, 'tcx> BodyCache<&'a mut Body<'tcx>> {
    #[inline]
    pub fn body(&self) -> &Body<'tcx> {
        self.body
    }

    #[inline]
    pub fn body_mut(&mut self) -> &mut Body<'tcx> {
        self.body
    }

    #[inline]
    pub fn basic_blocks(&self) -> &IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        &self.body.basic_blocks
    }

    #[inline]
    pub fn basic_blocks_mut(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        self.cache.basic_blocks_mut(&mut self.body)
    }
}

impl<'a, 'tcx> Deref for BodyCache<&'a mut Body<'tcx>> {
    type Target = Body<'tcx>;

    fn deref(&self) -> &Self::Target {
        self.body
    }
}

impl<'a, 'tcx> DerefMut for BodyCache<&'a mut Body<'tcx>> {
    fn deref_mut(&mut self) -> &mut Body<'tcx> {
        self.body
    }
}

impl<'a, 'tcx> Index<BasicBlock> for BodyCache<&'a mut Body<'tcx>> {
    type Output = BasicBlockData<'tcx>;

    #[inline]
    fn index(&self, index: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.body[index]
    }
}

impl<'a, 'tcx> IndexMut<BasicBlock> for BodyCache<&'a mut Body<'tcx>> {
    fn index_mut(&mut self, index: BasicBlock) -> &mut Self::Output {
        self.cache.invalidate_predecessors();
        &mut self.body.basic_blocks[index]
    }
}
