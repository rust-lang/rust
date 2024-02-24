// Regr. test that using HIR inlined from another krate does *not* add
// a dependency from the local hir_crate node. We can't easily test that
// directly anymore, so now we test that we get reuse.

//@ revisions: rpass1 rpass2
//@ compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "krate_inlined-x", cfg = "rpass2")]

fn main() {
    x::method();

    #[cfg(rpass2)]
    ()
}

mod x {
    pub fn method() {
        // use some methods that require inlining HIR from another crate:
        let mut v = vec![];
        v.push(1);
    }
}
