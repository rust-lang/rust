// This test makes sure that just changing a definition's location in the
// source file also changes its incr. comp. hash, if debuginfo is enabled.

// revisions:rpass1 rpass2 rpass3 rpass4

// ignore-asmjs wasm2js does not support source maps yet
// compile-flags: -g -Z query-dep-graph
// [rpass3]compile-flags: -Zincremental-relative-spans
// [rpass4]compile-flags: -Zincremental-relative-spans

#![feature(rustc_attrs)]
#![rustc_partition_codegened(module = "spans_significant_w_debuginfo", cfg = "rpass2")]
#![rustc_partition_codegened(module = "spans_significant_w_debuginfo", cfg = "rpass4")]

#[cfg(any(rpass1, rpass3))]
pub fn main() {}

#[cfg(any(rpass2, rpass4))]
#[rustc_clean(except = "hir_owner,hir_owner_nodes,optimized_mir", cfg = "rpass2")]
#[rustc_clean(cfg = "rpass4")]
pub fn main() {}
