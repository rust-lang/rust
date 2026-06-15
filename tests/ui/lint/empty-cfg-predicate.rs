//@ revisions: novers lowvers highvers
//@[lowvers] compile-flags: -Zhint-msrv=1.87.0
//@[highvers] compile-flags: -Zhint-msrv=1.89.0
//@[lowvers] check-pass

//! Check that we suggest `cfg(any())` -> `false` and `cfg(all())` -> true
//! Additionally tests the behaviour of empty cfg predicates.
#![deny(empty_cfg_predicate)]
#![crate_type = "lib"]

#[cfg(any())]
//[novers]~^ ERROR: use of empty `cfg(any())`
//[highvers]~^^ ERROR: use of empty `cfg(any())`
pub struct A;
#[cfg(all())]
//[novers]~^ ERROR: use of empty `cfg(all())`
//[highvers]~^^ ERROR: use of empty `cfg(all())`
pub struct B;

macro_rules! from_expansion {
    ($name:ident $(, $($meta:tt)*)?) => {
        #[cfg(any($($meta),*))]
        struct $name;
    }
}

from_expansion!(C);  // OK
