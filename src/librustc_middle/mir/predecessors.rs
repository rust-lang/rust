use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{Lock, LockGuard, MappedLockGuard};
use rustc_index::vec::IndexVec;
use rustc_serialize as serialize;
use smallvec::SmallVec;

use crate::mir::{BasicBlock, BasicBlockData};

// Typically 95%+ of basic blocks have 4 or fewer predecessors.
pub type Predecessors = IndexVec<BasicBlock, SmallVec<[BasicBlock; 4]>>;

#[derive(Clone, Debug)]
pub struct PredecessorCache {
    cache: Lock<Option<Predecessors>>,
}

impl PredecessorCache {
    #[inline]
    pub fn new() -> Self {
        PredecessorCache { cache: Lock::new(None) }
    }

    #[inline]
    pub fn invalidate(&mut self) {
        *self.cache.get_mut() = None;
    }

    #[inline]
    pub fn compute(
        &self,
        basic_blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>,
    ) -> MappedLockGuard<'_, Predecessors> {
        LockGuard::map(self.cache.lock(), |cache| {
            cache.get_or_insert_with(|| {
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
        })
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
