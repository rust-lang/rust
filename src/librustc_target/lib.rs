//! Some stuff used by rustc that doesn't have many dependencies
//!
//! Originally extracted from rustc::back, which was nominally the
//! compiler 'backend', though LLVM is rustc's backend, so rustc_target
//! is really just odds-and-ends relating to code gen and linking.
//! This crate mostly exists to make rustc smaller, so we might put
//! more 'stuff' here in the future. It does not have a dependency on
//! LLVM.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_syntax)]
#![feature(nll)]
#![feature(slice_patterns)]

#![deny(rust_2018_idioms)]
#![deny(unused_lifetimes)]

#[macro_use] extern crate log;

pub mod abi;
pub mod spec;
