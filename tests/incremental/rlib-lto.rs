//@ revisions: bpass1 bpass2
//@ compile-flags: -Z query-dep-graph --crate-type rlib -C lto

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "rlib_lto", cfg = "bpass2")]

pub fn foo() {}
