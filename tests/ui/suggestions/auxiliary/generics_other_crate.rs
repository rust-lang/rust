//! This file is used to test suggestions for generics in other crates.

#![allow(unused_unconstructable_pub_structs)]

pub struct External;
pub struct ExternalGeneric<T>(T);
