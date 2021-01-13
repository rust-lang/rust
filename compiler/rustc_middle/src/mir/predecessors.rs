//! Lazily compute the reverse control-flow graph for the MIR.

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::OnceCell;
use rustc_index::vec::IndexVec;
use rustc_serialize as serialize;
use smallvec::SmallVec;

use crate::mir::{BasicBlock, BasicBlockData};

// Typically 95%+ of basic blocks have 4 or fewer predecessors.
pub type Predecessors = IndexVec<BasicBlock, SmallVec<[BasicBlock; 4]>>;

#[derive(Clone, Debug)]
pub(super) struct PredecessorCache {
    cache: OnceCell<Predecessors>,
}

impl PredecessorCache {
    #[inline]
    pub(super) fn new() -> Self {
        PredecessorCache { cache: OnceCell::new() }
    }

    /// Invalidates the predecessor cache.
    #[inline]
    pub(super) fn invalidate(&mut self) {
        // Invalidating the predecessor cache requires mutating the MIR, which in turn requires a
        // unique reference (`&mut`) to the `mir::Body`. Because of this, we can assume that all
        // callers of `invalidate` have a unique reference to the MIR and thus to the predecessor
        // cache. This means we never need to do synchronization when `invalidate` is called, we can
        // simply reinitialize the `OnceCell`.
        self.cache = OnceCell::new();
    }

    /// Returns the predecessor graph for this MIR.
    #[inline]
    pub(super) fn compute(
        &self,
        basic_blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>,
    ) -> &Predecessors {
        self.cache.get_or_init(|| {
            let mut preds = IndexVec::from_elem(SmallVec::new(), basic_blocks);
            for (bb, data) in basic_blocks.iter_enumerated() {
                if let Some(term) = &data.terminator {
                    for &succ in term.successors() {
                        preds[succ].push(bb);
                    }
                }
            }

            preds
        })
    }
}

impl<S: serialize::Encoder> serialize::Encodable<S> for PredecessorCache {
    #[inline]
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        serialize::Encodable::encode(&(), s)
    }
}

impl<D: serialize::Decoder> serialize::Decodable<D> for PredecessorCache {
    #[inline]
    fn decode(d: &mut D) -> Result<Self, D::Error> {
        serialize::Decodable::decode(d).map(|_v: ()| Self::new())
    }
}

impl<CTX> HashStable<CTX> for PredecessorCache {
    #[inline]
    fn hash_stable(&self, _: &mut CTX, _: &mut StableHasher) {
        // do nothing
    }
}

TrivialTypeFoldableAndLiftImpls! {
    PredecessorCache,
}
