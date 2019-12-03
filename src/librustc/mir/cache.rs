use rustc_index::vec::IndexVec;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_serialize::{Encodable, Encoder, Decodable, Decoder};
use crate::ich::StableHashingContext;
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

impl rustc_serialize::Encodable for Cache {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        Encodable::encode(&(), s)
    }
}

impl rustc_serialize::Decodable for Cache {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        Decodable::decode(d).map(|_v: ()| Self::new())
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for Cache {
    fn hash_stable(&self, _: &mut StableHashingContext<'a>, _: &mut StableHasher) {
        // Do nothing.
    }
}

impl Cache {
    pub fn new() -> Self {
        Self {
            predecessors: None,
        }
    }

    pub fn invalidate_predecessors(&mut self) {
        // FIXME: consider being more fine-grained
        self.predecessors = None;
    }

    pub fn ensure_predecessors(&mut self, body: &Body<'_>) {
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
    }

    /// This will recompute the predecessors cache if it is not available
    fn predecessors(&mut self, body: &Body<'_>) -> &IndexVec<BasicBlock, Vec<BasicBlock>> {
        self.ensure_predecessors(body);
        self.predecessors.as_ref().unwrap()
    }

    fn unwrap_predecessors_for(&self, bb: BasicBlock) -> &[BasicBlock] {
        &self.predecessors.as_ref().unwrap()[bb]
    }

    fn unwrap_predecessor_locations<'a>(
        &'a self,
        loc: Location,
        body: &'a Body<'a>
    ) -> impl Iterator<Item = Location> + 'a {
        let if_zero_locations = if loc.statement_index == 0 {
            let predecessor_blocks = self.unwrap_predecessors_for(loc.block);
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

    pub fn basic_blocks_mut<'a, 'tcx>(
        &mut self,
        body: &'a mut Body<'tcx>
    ) -> &'a mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        debug!("bbm: Clearing predecessors cache for body at: {:?}", body.span.data());
        self.invalidate_predecessors();
        &mut body.basic_blocks
    }

    pub fn basic_blocks_and_local_decls_mut<'a, 'tcx>(
        &mut self,
        body: &'a mut Body<'tcx>
    ) -> (&'a mut IndexVec<BasicBlock, BasicBlockData<'tcx>>, &'a mut LocalDecls<'tcx>) {
        debug!("bbaldm: Clearing predecessors cache for body at: {:?}", body.span.data());
        self.invalidate_predecessors();
        (&mut body.basic_blocks, &mut body.local_decls)
    }
}

#[derive(Clone, Debug, HashStable, RustcEncodable, RustcDecodable, TypeFoldable)]
pub struct BodyCache<'tcx> {
    cache: Cache,
    body: Body<'tcx>,
}

impl BodyCache<'tcx> {
    pub fn new(body: Body<'tcx>) -> Self {
        Self {
            cache: Cache::new(),
            body,
        }
    }
}

#[macro_export]
macro_rules! read_only {
    ($body:expr) => {
        {
            $body.ensure_predecessors();
            $body.unwrap_read_only()
        }
    };
}

impl BodyCache<'tcx> {
    pub fn ensure_predecessors(&mut self) {
        self.cache.ensure_predecessors(&self.body);
    }

    pub fn predecessors(&mut self) -> &IndexVec<BasicBlock, Vec<BasicBlock>> {
        self.cache.predecessors(&self.body)
    }

    pub fn unwrap_read_only(&self) -> ReadOnlyBodyCache<'_, 'tcx> {
        ReadOnlyBodyCache::new(&self.cache, &self.body)
    }

    pub fn basic_blocks_mut(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        self.cache.basic_blocks_mut(&mut self.body)
    }

    pub fn basic_blocks_and_local_decls_mut(
        &mut self
    ) -> (&mut IndexVec<BasicBlock, BasicBlockData<'tcx>>, &mut LocalDecls<'tcx>) {
        self.cache.basic_blocks_and_local_decls_mut(&mut self.body)
    }
}

impl<'tcx> Index<BasicBlock> for BodyCache<'tcx> {
    type Output = BasicBlockData<'tcx>;

    fn index(&self, index: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.body[index]
    }
}

impl<'tcx> IndexMut<BasicBlock> for BodyCache<'tcx> {
    fn index_mut(&mut self, index: BasicBlock) -> &mut Self::Output {
        &mut self.basic_blocks_mut()[index]
    }
}

impl<'tcx> Deref for BodyCache<'tcx> {
    type Target = Body<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.body
    }
}

impl<'tcx> DerefMut for BodyCache<'tcx> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.body
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ReadOnlyBodyCache<'a, 'tcx> {
    cache: &'a Cache,
    body: &'a Body<'tcx>,
}

impl ReadOnlyBodyCache<'a, 'tcx> {
    fn new(cache: &'a Cache, body: &'a Body<'tcx>) -> Self {
        assert!(
            cache.predecessors.is_some(),
            "Cannot construct ReadOnlyBodyCache without computed predecessors");
        Self {
            cache,
            body,
        }
    }

    pub fn predecessors(&self) -> &IndexVec<BasicBlock, Vec<BasicBlock>> {
        self.cache.predecessors.as_ref().unwrap()
    }

    pub fn predecessors_for(&self, bb: BasicBlock) -> &[BasicBlock] {
        self.cache.unwrap_predecessors_for(bb)
    }

    pub fn predecessor_locations(&self, loc: Location) -> impl Iterator<Item = Location> + '_ {
        self.cache.unwrap_predecessor_locations(loc, self.body)
    }

    pub fn body(&self) -> &'a Body<'tcx> {
        self.body
    }

    pub fn basic_blocks(&self) -> &IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        &self.body.basic_blocks
    }

    pub fn dominators(&self) -> Dominators<BasicBlock> {
        dominators(self)
    }
}

impl graph::DirectedGraph for ReadOnlyBodyCache<'a, 'tcx> {
    type Node = BasicBlock;
}

impl graph::GraphPredecessors<'graph> for ReadOnlyBodyCache<'a, 'tcx> {
    type Item = BasicBlock;
    type Iter = IntoIter<BasicBlock>;
}

impl graph::WithPredecessors for ReadOnlyBodyCache<'a, 'tcx> {
    fn predecessors(
        &self,
        node: Self::Node,
    ) -> <Self as GraphPredecessors<'_>>::Iter {
        self.cache.unwrap_predecessors_for(node).to_vec().into_iter()
    }
}

impl graph::WithNumNodes for ReadOnlyBodyCache<'a, 'tcx> {
    fn num_nodes(&self) -> usize {
        self.body.num_nodes()
    }
}

impl graph::WithStartNode for ReadOnlyBodyCache<'a, 'tcx> {
    fn start_node(&self) -> Self::Node {
        self.body.start_node()
    }
}

impl graph::WithSuccessors for ReadOnlyBodyCache<'a, 'tcx> {
    fn successors(
        &self,
        node: Self::Node,
    ) -> <Self as GraphSuccessors<'_>>::Iter {
        self.body.successors(node)
    }
}

impl<'a, 'b, 'tcx> graph::GraphSuccessors<'b> for ReadOnlyBodyCache<'a, 'tcx> {
    type Item = BasicBlock;
    type Iter = iter::Cloned<Successors<'b>>;
}


impl Deref for ReadOnlyBodyCache<'a, 'tcx> {
    type Target = &'a Body<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.body
    }
}

CloneTypeFoldableAndLiftImpls! {
    Cache,
}
