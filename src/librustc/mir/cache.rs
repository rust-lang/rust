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

macro_rules! get_predecessors {
    (mut $self:ident, $block:expr, $body:expr) => {
        $self.predecessors_for($block, $body)
    };
    ($self:ident, $block:expr, $body:expr) => {
        $self.unwrap_predecessors_for($block)
    };
}

macro_rules! impl_predecessor_locations {
    ( ( $($pub:ident)? )  $name:ident $($mutability:ident)?) => {
        $($pub)? fn $name<'a>(&'a $($mutability)? self, loc: Location, body: &'a Body<'a>) -> impl Iterator<Item = Location> + 'a {
            let if_zero_locations = if loc.statement_index == 0 {
                let predecessor_blocks = get_predecessors!($($mutability)? self, loc.block, body);
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
    };
}

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

    #[inline]
    /// This will recompute the predecessors cache if it is not available
    pub fn predecessors(&mut self, body: &Body<'_>) -> &IndexVec<BasicBlock, Vec<BasicBlock>> {
        self.ensure_predecessors(body);
        self.predecessors.as_ref().unwrap()
    }

    #[inline]
    pub fn predecessors_for(&mut self, bb: BasicBlock, body: &Body<'_>) -> &[BasicBlock] {
        &self.predecessors(body)[bb]
    }

    #[inline]
    fn unwrap_predecessors_for(&self, bb: BasicBlock) -> &[BasicBlock] {
        &self.predecessors.as_ref().unwrap()[bb]
    }

    #[inline]
    impl_predecessor_locations!((pub) predecessor_locations mut);

    impl_predecessor_locations!(() unwrap_predecessor_locations);

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
    pub fn read_only(mut self) -> ReadOnlyBodyCache<'a, 'tcx> {
        self.cache.ensure_predecessors(self.body);
        ReadOnlyBodyCache {
            cache: self.cache,
            body: self.body,
        }
    }

    #[inline]
    pub fn basic_blocks(&self) -> &IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        &self.body.basic_blocks
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

pub struct ReadOnlyBodyCache<'a, 'tcx> {
    cache: Cache,
    body: &'a Body<'tcx>,
}

impl ReadOnlyBodyCache<'a, 'tcx> {
    #[inline]
    pub fn predecessors_for(&self, bb: BasicBlock) -> &[BasicBlock] {
        self.cache.unwrap_predecessors_for(bb)
    }

    #[inline]
    pub fn predecessor_locations(&self, loc: Location) -> impl Iterator<Item = Location> + '_ {
        self.cache.unwrap_predecessor_locations(loc, self.body)
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
    pub fn dominators(&self) -> Dominators<BasicBlock> {
        dominators(self)
    }

    pub fn to_owned(self) -> BodyCache<&'a Body<'tcx>> {
        BodyCache {
            cache: self.cache,
            body: self.body,
        }
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
    type Target = Body<'tcx>;

    fn deref(&self) -> &Self::Target {
        self.body
    }
}

impl Index<BasicBlock> for ReadOnlyBodyCache<'a, 'tcx> {
    type Output = BasicBlockData<'tcx>;

    #[inline]
    fn index(&self, index: BasicBlock) -> &BasicBlockData<'tcx> {
        &self.body[index]
    }
}