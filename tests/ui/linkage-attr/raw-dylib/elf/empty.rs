//@ only-x86_64-unknown-linux-gnu
//@ needs-dynamic-linking
//@ build-pass

#![allow(incomplete_features)]
#![feature(raw_dylib_elf)]

#[link(name = "hack", kind = "raw-dylib")]
unsafe extern "C" {}

fn main() {}
