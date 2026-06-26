//! Implementation of bootstrap, the Rust build system.
//!
//! This module, and its descendants, are the implementation of the Rust build
//! system. Most of this build system is backed by Cargo but the outer layer
//! here serves as the ability to orchestrate calling Cargo, sequencing Cargo
//! builds, building artifacts like LLVM, etc. The goals of bootstrap are:
//!
//! * To be an easily understandable, easily extensible, and maintainable build
//!   system.
//! * Leverage standard tools in the Rust ecosystem to build the compiler, aka
//!   crates.io and Cargo.
//! * A standard interface to build across all platforms, including MSVC
//!
//! ## Further information
//!
//! More documentation can be found in each respective module below, and you can
//! also check out the `src/bootstrap/README.md` file for more information.

// tidy-alphabetical-start
#![allow(clippy::assertions_on_constants, reason = "false positive for `assert!(cfg!(..))`")]
#![allow(clippy::map_clone, reason = "false positive for `|x: &&Foo| Foo::clone(x)`")]
// tidy-alphabetical-end

pub mod cli_main;
mod core;
mod utils;
