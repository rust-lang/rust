//@ aux-crate:priv:private_dep=private-dep.rs
//@ compile-flags: -Zunstable-options

extern crate private_dep;
use private_dep::A;

pub struct B;

impl A for B {
    fn foo() {}
}
