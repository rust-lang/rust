//@ revisions: bfail1 bfail2
//@ compile-flags: -Z query-dep-graph --crate-type rlib -C linker-plugin-lto -O
//@ no-prefer-dynamic
//@ build-pass

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "lto_in_linker", cfg = "bfail2")]

pub fn foo() {}
