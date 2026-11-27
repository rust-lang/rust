//@ aux-build:sealed-trait-foreign-indirect-aux.rs

extern crate sealed_trait_foreign_indirect_aux as indirect;

mod private {
    pub trait Hidden {}
}

pub trait Sealed: private::Hidden {}
pub trait NotSealed: indirect::Super {}
