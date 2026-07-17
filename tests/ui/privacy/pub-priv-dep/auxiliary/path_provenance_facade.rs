//@ aux-crate:path_provenance_leaf=path_provenance_leaf.rs

#![feature(trait_alias)]

extern crate path_provenance_leaf as leaf;

pub mod nested {
    pub use crate::leaf::{T, U};
}

pub use leaf::{Const, N, T, Tr, U};

pub type Alias = leaf::T;
pub type Generic<X> = (leaf::T, X);
pub type Defaulted<X = leaf::T> = X;
pub struct Nominal<X = leaf::T>(pub X);
pub type ConstDefault<const N: usize = { leaf::N }> = leaf::Const<N>;
pub type Primitive = ();
pub trait TraitAlias = leaf::Tr;
