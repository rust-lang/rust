use rustc_index::vec::IndexVec;
use rustc_data_structures::sync::{RwLock, MappedReadGuard, ReadGuard};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_serialize::{Encodable, Encoder, Decodable, Decoder};
use crate::ich::StableHashingContext;
use crate::mir::{Body, BasicBlock};

#[derive(Clone, Debug)]
pub struct Cache {
    predecessors: RwLock<Option<IndexVec<BasicBlock, Vec<BasicBlock>>>>
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
        Cache {
            predecessors: RwLock::new(None)
        }
    }

    pub fn invalidate(&self) {
        // FIXME: consider being more fine-grained
        *self.predecessors.borrow_mut() = None;
    }

    pub fn predecessors(
        &self,
        body: &Body<'_>
    ) -> MappedReadGuard<'_, IndexVec<BasicBlock, Vec<BasicBlock>>> {
        if self.predecessors.borrow().is_none() {
            *self.predecessors.borrow_mut() = Some(calculate_predecessors(body));
        }

        ReadGuard::map(self.predecessors.borrow(), |p| p.as_ref().unwrap())
    }
}

fn calculate_predecessors(body: &Body<'_>) -> IndexVec<BasicBlock, Vec<BasicBlock>> {
    let mut result = IndexVec::from_elem(vec![], body.basic_blocks());
    for (bb, data) in body.basic_blocks().iter_enumerated() {
        if let Some(ref term) = data.terminator {
            for &tgt in term.successors() {
                result[tgt].push(bb);
            }
        }
    }

    result
}

CloneTypeFoldableAndLiftImpls! {
    Cache,
}
