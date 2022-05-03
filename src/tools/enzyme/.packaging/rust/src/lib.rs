#![deny(missing_docs)]

//! This crate which helps to build and use [Enzyme](https://enzyme.mit.edu) from Rust.
//!
//! ## Usage
//!
//! First, install everything required using:
//! ```Bash
//! $ cargo install enzyme
//! $ install-enzyme
//! ````
//! The second command might take a few hours, depending on you cpu.
//!
//! Afterwards, you can compile a crate using [oxide-enzyme](https://github.com/rust-ml/oxide-enzyme)
//! with the command
//! ```Bash
//! $ cargo enzyme
//! ```
//! This is equivalent to a `cargo build` call in crates not usinng Enzyme.
//! We currently don't support other configurations, extra parameters will be ignored.
//!
//!
//! ## Goals
//!
//! The goal of this crate is to simplify experimenting with Enzyme in Rust.
//! We are already working on a new iteration which will not require an extra setup step
//! and will support arbitrary Rust code.
//! So please feel free to give feedback and raise issues in our Github, but keep in mind
//! that the will focus on the new oxide-enzyme iteration, rather than fixing all bugs in this one.
//!
//! ## Other Languages
//! C/C++ support is available [here](https://github.com/wsmoses/Enzyme).  
//! Julia support is available [here](https://github.com/wsmoses/Enzyme.jl).  
//! A code explorer instance for C++ is available [here](https://enzyme.mit.edu/explorer).

mod code;

pub use code::compile::build;
pub use code::downloader::download;
pub use code::generate_api::*;
pub use code::utils::{get_bindings_path, Cli, Repo};
