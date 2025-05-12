// Test that debuginfo does not introduce a dependency edge to the hir_crate
// dep-node.

//@ revisions:rpass1 rpass2
//@ compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "issue_38222-mod1", cfg = "rpass2")]
// If codegen had added a dependency edge to the hir_crate dep-node, nothing would
// be re-used, so checking that this module was re-used is sufficient.
#![rustc_partition_reused(module = "issue_38222", cfg = "rpass2")]

//@[rpass1] compile-flags: -C debuginfo=1
//@[rpass2] compile-flags: -C debuginfo=1

pub fn main() {
    mod1::some_fn();
}

mod mod1 {
    pub fn some_fn() {
        #[cfg(rpass2)]
        {}

        let _ = 1;
    }
}
