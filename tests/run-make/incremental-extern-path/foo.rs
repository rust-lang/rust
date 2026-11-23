#![feature(no_core, rustc_attrs)]
#![no_core]
#![no_main]
#![rustc_partition_reused(module = "repro", cfg = "reuse")]

extern crate minicore;

#[unsafe(no_mangle)]
pub fn entry() {}
