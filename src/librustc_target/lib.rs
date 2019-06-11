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
#![feature(rustc_attrs)]
#![feature(slice_patterns)]
#![feature(step_trait)]

#![deny(rust_2018_idioms)]
#![deny(internal)]
#![deny(unused_lifetimes)]

#[macro_use] extern crate log;

#[allow(unused_extern_crates)]
extern crate serialize as rustc_serialize; // used by deriving

// See librustc_cratesio_shim/Cargo.toml for a comment explaining this.
#[allow(unused_extern_crates)]
extern crate rustc_cratesio_shim;

#[macro_use]
extern crate rustc_data_structures;

pub mod abi;
pub mod spec;
