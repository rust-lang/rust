use rustc_index::vec::IndexVec;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_serialize::{Encodable, Encoder, Decodable, Decoder};
use crate::ich::StableHashingContext;
use crate::mir::BasicBlock;

#[derive(Clone, Debug)]
pub(in crate::mir) struct Cache {
    pub(in crate::mir) predecessors: Option<IndexVec<BasicBlock, Vec<BasicBlock>>>
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
            predecessors: None
        }
    }

    #[inline]
    pub fn invalidate_predecessors(&mut self) {
        // FIXME: consider being more fine-grained
        self.predecessors = None;
    }
}

CloneTypeFoldableAndLiftImpls! {
    Cache,
}
