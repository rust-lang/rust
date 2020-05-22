//! Lazily compute the reverse control-flow graph for the MIR.

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{Lock, Lrc};
use rustc_index::vec::IndexVec;
use rustc_serialize as serialize;
use smallvec::SmallVec;

use crate::mir::{BasicBlock, BasicBlockData};

// Typically 95%+ of basic blocks have 4 or fewer predecessors.
pub type Predecessors = IndexVec<BasicBlock, SmallVec<[BasicBlock; 4]>>;

#[derive(Clone, Debug)]
pub(super) struct PredecessorCache {
    cache: Lock<Option<Lrc<Predecessors>>>,
}

impl PredecessorCache {
    #[inline]
    pub(super) fn new() -> Self {
        PredecessorCache { cache: Lock::new(None) }
    }

    /// Invalidates the predecessor cache.
    ///
    /// Invalidating the predecessor cache requires mutating the MIR, which in turn requires a
    /// unique reference (`&mut`) to the `mir::Body`. Because of this, we can assume that all
    /// callers of `invalidate` have a unique reference to the MIR and thus to the predecessor
    /// cache. This means we don't actually need to take a lock when `invalidate` is called.
    #[inline]
    pub(super) fn invalidate(&mut self) {
        *self.cache.get_mut() = None;
    }

    /// Returns a ref-counted smart pointer containing the predecessor graph for this MIR.
    ///
    /// We use ref-counting instead of a mapped `LockGuard` here to ensure that the lock for
    /// `cache` is only held inside this function. As long as no other locks are taken while
    /// computing the predecessor graph, deadlock is impossible.
    #[inline]
    pub(super) fn compute(
        &self,
        basic_blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>,
    ) -> Lrc<Predecessors> {
        Lrc::clone(self.cache.lock().get_or_insert_with(|| {
            let mut preds = IndexVec::from_elem(SmallVec::new(), basic_blocks);
            for (bb, data) in basic_blocks.iter_enumerated() {
                if let Some(term) = &data.terminator {
                    for &succ in term.successors() {
                        preds[succ].push(bb);
                    }
                }
            }

            Lrc::new(preds)
        }))
    }
}

impl serialize::Encodable for PredecessorCache {
    #[inline]
    fn encode<S: serialize::Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        serialize::Encodable::encode(&(), s)
    }
}

impl serialize::Decodable for PredecessorCache {
    #[inline]
    fn decode<D: serialize::Decoder>(d: &mut D) -> Result<Self, D::Error> {
        serialize::Decodable::decode(d).map(|_v: ()| Self::new())
    }
}

impl<CTX> HashStable<CTX> for PredecessorCache {
    #[inline]
    fn hash_stable(&self, _: &mut CTX, _: &mut StableHasher) {
        // do nothing
    }
}

CloneTypeFoldableAndLiftImpls! {
    PredecessorCache,
}
