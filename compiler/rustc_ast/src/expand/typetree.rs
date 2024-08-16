use std::fmt;

use crate::expand::{Decodable, Encodable, HashStable_Generic};

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

#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct TypeTree(pub Vec<Type>);

impl TypeTree {
    pub fn new() -> Self {
        Self(Vec::new())
    }
    pub fn all_ints() -> Self {
        Self(vec![Type { offset: -1, size: 1, kind: Kind::Integer, child: TypeTree::new() }])
    }
    pub fn int(size: usize) -> Self {
        let mut ints = Vec::with_capacity(size);
        for i in 0..size {
            ints.push(Type {
                offset: i as isize,
                size: 1,
                kind: Kind::Integer,
                child: TypeTree::new(),
            });
        }
        Self(ints)
    }
}

#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct FncTree {
    pub args: Vec<TypeTree>,
    pub ret: TypeTree,
}

#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct Type {
    pub offset: isize,
    pub size: usize,
    pub kind: Kind,
    pub child: TypeTree,
}

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
