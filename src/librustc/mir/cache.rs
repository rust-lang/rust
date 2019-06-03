use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::sync::{RwLock, MappedReadGuard, ReadGuard};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};
use crate::ich::StableHashingContext;
use crate::mir::{Body, BasicBlock};

use crate::rustc_serialize as serialize;

#[derive(Clone, Debug)]
pub struct Cache {
    predecessors: RwLock<Option<IndexVec<BasicBlock, Vec<BasicBlock>>>>
}


impl serialize::Encodable for Cache {
    fn encode<S: serialize::Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        serialize::Encodable::encode(&(), s)
    }
}

impl serialize::Decodable for Cache {
    fn decode<D: serialize::Decoder>(d: &mut D) -> Result<Self, D::Error> {
        serialize::Decodable::decode(d).map(|_v: ()| Self::new())
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for Cache {
    fn hash_stable<W: StableHasherResult>(&self,
                                          _: &mut StableHashingContext<'a>,
                                          _: &mut StableHasher<W>) {
        // do nothing
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
