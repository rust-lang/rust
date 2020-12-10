//! Some stuff used by rustc that doesn't have many dependencies
//!
//! Originally extracted from rustc::back, which was nominally the
//! compiler 'backend', though LLVM is rustc's backend, so rustc_target
//! is really just odds-and-ends relating to code gen and linking.
//! This crate mostly exists to make rustc smaller, so we might put
//! more 'stuff' here in the future. It does not have a dependency on
//! LLVM.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(bool_to_option)]
#![feature(const_fn)]
#![feature(const_panic)]
#![feature(nll)]
#![feature(never_type)]
#![feature(associated_type_bounds)]
#![feature(exhaustive_patterns)]
#![feature(str_split_once)]

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate tracing;

pub mod abi;
pub mod asm;
pub mod spec;

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in librustc_middle.
pub trait HashStableContext {}
