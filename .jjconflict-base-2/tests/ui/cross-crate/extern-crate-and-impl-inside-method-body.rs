//@ run-pass
//@ aux-build:extern-crate-and-impl-inside-method-body.rs

//! Regression test for https://github.com/rust-lang/rust/issues/41053
//! An extern crate declaration and trait impl referencing that crate
//! inside an impl method body triggered a borrow conflict in the
//! compiler's visible_parent_map.

#![allow(non_local_definitions)]

pub trait Trait { fn foo(&self) {} }

pub struct Foo;

impl Iterator for Foo {
    type Item = Box<dyn Trait>;
    fn next(&mut self) -> Option<Box<dyn Trait>> {
        extern crate extern_crate_and_impl_inside_method_body as issue_41053;
        impl crate::Trait for issue_41053::Test {
            fn foo(&self) {}
        }
        Some(Box::new(issue_41053::Test))
    }
}

fn main() {
    Foo.next().unwrap().foo();
}
