// revisions: cfail1 cfail2
// compile-flags: -Z query-dep-graph
// build-pass (FIXME(62277): could be check-pass?)

#![allow(warnings)]
#![feature(rustc_attrs)]
#![rustc_partition_reused(module="krate_inherent-x", cfg="cfail2")]
#![crate_type = "rlib"]

pub mod x {
    pub struct Foo;
    impl Foo {
        pub fn foo(&self) { }
    }

    pub fn method() {
        let x: Foo = Foo;
        x.foo(); // inherent methods used to add an edge from Krate
    }
}

#[cfg(cfail1)]
pub fn bar() { } // remove this unrelated fn in cfail2, which should not affect `x::method`
