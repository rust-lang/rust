//@ only-apple
//@ build-pass

#![allow(incomplete_features)]
#![feature(raw_dylib_macho)]

#[link(name = "hack", kind = "raw-dylib", modifiers = "+verbatim")]
unsafe extern "C" {}

fn main() {}
