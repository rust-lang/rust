//@ revisions: bfail1 bfail2
//@ compile-flags: -Z query-dep-graph
//@ build-pass (FIXME(62277): could be check-pass?)
//@ ignore-backends: gcc

#![allow(warnings)]
#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "krate_inherent-x", cfg = "bfail2")]
#![crate_type = "rlib"]

pub mod x {
    pub struct Foo;
    impl Foo {
        pub fn foo(&self) {}
    }

    pub fn method() {
        let x: Foo = Foo;
        x.foo(); // inherent methods used to add an edge from hir_crate
    }
}

#[cfg(bfail1)]
pub fn bar() {} // remove this unrelated fn in bfail2, which should not affect `x::method`
