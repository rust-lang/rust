// This test makes sure that just changing a definition's location in the
// source file also changes its incr. comp. hash, if debuginfo is enabled.

//@ revisions:rpass1 rpass2

//@ compile-flags: -g -Z query-dep-graph

#![feature(rustc_attrs)]
#![rustc_partition_codegened(module = "spans_significant_w_debuginfo", cfg = "rpass2")]

#[cfg(rpass1)]
pub fn main() {}

#[cfg(rpass2)]
#[rustc_clean(cfg = "rpass2")]
pub fn main() {}
