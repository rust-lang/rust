use std::fmt;
//use rustc_data_structures::stable_hasher::{HashStable};//, StableHasher};
//use crate::HashStableContext;


#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum Kind {
    Anything,
    Integer,
    Pointer,
    Half,
    Float,
    Double,
    Unknown,
}
//impl<CTX: HashStableContext> HashStable<CTX> for Kind {
//    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
//        clause_kind_discriminant(self).hash_stable(hcx, hasher);
//    }
//}
//fn clause_kind_discriminant(value: &Kind) -> usize {
//    match value {
//        Kind::Anything => 0,
//        Kind::Integer => 1,
//        Kind::Pointer => 2,
//        Kind::Half => 3,
//        Kind::Float => 4,
//        Kind::Double => 5,
//        Kind::Unknown => 6,
//    }
//}

#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct TypeTree(pub Vec<Type>);

#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct Type {
    pub offset: isize,
    pub size: usize,
    pub kind: Kind,
    pub child: TypeTree,
}

//impl<CTX: HashStableContext> HashStable<CTX> for Type {
//    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
//        self.offset.hash_stable(hcx, hasher);
//        self.size.hash_stable(hcx, hasher);
//        self.kind.hash_stable(hcx, hasher);
//        self.child.0.hash_stable(hcx, hasher);
//    }
//}

impl Type {
    pub fn add_offset(self, add: isize) -> Self {
        let offset = match self.offset {
            -1 => add,
            x => add + x,
        };

        Self { size: self.size, kind: self.kind, child: self.child, offset }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Debug>::fmt(self, f)
    }
}
