//! This module contains the definition of the `TypeTree` and `Type` structs.
//! They are thin Rust wrappers around the TypeTrees used by Enzyme as the LLVM based autodiff
//! backend. The Enzyme TypeTrees currently have various limitations and should be rewritten, so the
//! Rust frontend obviously has the same limitations. The main motivation of TypeTrees is to
//! represent how a type looks like "in memory". Enzyme can deduce this based on usage patterns in
//! the user code, but this is extremely slow and not even always sufficient. As such we lower some
//! information from rustc to help Enzyme. For a full explanation of their design it is necessary to
//! analyze the implementation in Enzyme core itself. As a rough summary, `-1` in Enzyme speech means
//! everywhere. That is `{0:-1: Float}` means at index 0 you have a ptr, if you dereference it it
//! will be floats everywhere. Thus `* f32`. If you have `{-1:int}` it means int's everywhere,
//! e.g. [i32; N]. `{0:-1:-1 float}` then means one pointer at offset 0, if you dereference it there
//! will be only pointers, if you dereference these new pointers they will point to array of floats.
//! Generally, it allows byte-specific descriptions.
//! FIXME: This description might be partly inaccurate and should be extended, along with
//! adding documentation to the corresponding Enzyme core code.
//! FIXME: Rewrite the TypeTree logic in Enzyme core to reduce the need for the rustc frontend to
//! provide typetree information.
//! FIXME: We should also re-evaluate where we create TypeTrees from Rust types, since MIR
//! representations of some types might not be accurate. For example a vector of floats might be
//! represented as a vector of u8s in MIR in some cases.

use std::fmt;

use crate::expand::{Decodable, Encodable, HashStable_Generic};

#[derive(Clone, Copy, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum Kind {
    Anything,
    Integer,
    Pointer,
    Half,
    Float,
    Double,
    F128,
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
