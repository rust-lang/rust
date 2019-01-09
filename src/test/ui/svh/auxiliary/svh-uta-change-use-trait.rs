//! "compile-fail/svh-uta-trait.rs" is checking that we detect a
//! change from `use foo::TraitB` to use `foo::TraitB` in the hash
//! (SVH) computation (#14132), since that will affect method
//! resolution.
//!
//! This is the upstream crate.

#![crate_name = "uta"]

mod traits {
    pub trait TraitA { fn val(&self) -> isize { 2 } }
    pub trait TraitB { fn val(&self) -> isize { 3 } }
}

impl traits::TraitA for () {}
impl traits::TraitB for () {}

pub fn foo<T>(_: isize) -> isize {
    use traits::TraitB;
    let v = ();
    v.val()
}
