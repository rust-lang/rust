//! Implements `Eq` for vector types.
#![allow(unused)]

macro_rules! impl_eq {
    ($id:ident) => { impl ::cmp::Eq for $id {} }
}
