use rustc_data_structures::graph::{
    self, DirectedGraph, WithNumNodes, WithStartNode, WithSuccessors,
};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::OnceCell;
use rustc_serialize as serialize;

/// Helper type to cache the result of `graph::is_cyclic`.
#[derive(Clone, Debug)]
pub(super) struct GraphIsCyclicCache {
    cache: OnceCell<bool>,
}

impl GraphIsCyclicCache {
    #[inline]
    pub(super) fn new() -> Self {
        GraphIsCyclicCache { cache: OnceCell::new() }
    }

    pub(super) fn is_cyclic<G>(&self, graph: &G) -> bool
    where
        G: ?Sized + DirectedGraph + WithStartNode + WithSuccessors + WithNumNodes,
    {
        *self.cache.get_or_init(|| graph::is_cyclic(graph))
    }

    /// Invalidates the cache.
    #[inline]
    pub(super) fn invalidate(&mut self) {
        // Invalidating the cache requires mutating the MIR, which in turn requires a unique
        // reference (`&mut`) to the `mir::Body`. Because of this, we can assume that all
        // callers of `invalidate` have a unique reference to the MIR and thus to the
        // cache. This means we never need to do synchronization when `invalidate` is called,
        // we can simply reinitialize the `OnceCell`.
        self.cache = OnceCell::new();
    }
}

impl<S: serialize::Encoder> serialize::Encodable<S> for GraphIsCyclicCache {
    #[inline]
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        serialize::Encodable::encode(&(), s)
    }
}

impl<D: serialize::Decoder> serialize::Decodable<D> for GraphIsCyclicCache {
    #[inline]
    fn decode(d: &mut D) -> Result<Self, D::Error> {
        serialize::Decodable::decode(d).map(|_v: ()| Self::new())
    }
}

impl<CTX> HashStable<CTX> for GraphIsCyclicCache {
    #[inline]
    fn hash_stable(&self, _: &mut CTX, _: &mut StableHasher) {
        // do nothing
    }
}

TrivialTypeFoldableAndLiftImpls! {
    GraphIsCyclicCache,
}
