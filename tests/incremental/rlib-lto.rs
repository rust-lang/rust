//@ revisions:cfail1 cfail2
//@ compile-flags: -Zretain-dep-graph --crate-type rlib -C lto
//@ build-pass

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "rlib_lto", cfg = "cfail2")]

pub fn foo() {}
