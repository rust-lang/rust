// This test makes sure that just changing a definition's location in the
// source file also changes its incr. comp. hash, if debuginfo is enabled.

// revisions:rpass1 rpass2 rpass3 rpass4
// [rpass3]compile-flags: -Zincremental-relative-spans
// [rpass4]compile-flags: -Zincremental-relative-spans

// compile-flags: -C overflow-checks=on -Z query-dep-graph

#![feature(rustc_attrs)]
#![feature(bench_black_box)]
#![rustc_partition_codegened(module = "spans_significant_w_panic", cfg = "rpass2")]
#![rustc_partition_codegened(module = "spans_significant_w_panic", cfg = "rpass4")]

#[cfg(any(rpass1, rpass3))]
pub fn main() {
    if std::hint::black_box(false) {
        panic!()
    }
}

#[cfg(any(rpass2, rpass4))]
#[rustc_clean(except = "hir_owner,hir_owner_nodes,optimized_mir", cfg = "rpass2")]
#[rustc_clean(cfg = "rpass4")]
pub fn main() {
    if std::hint::black_box(false) {
        panic!()
    }
}
