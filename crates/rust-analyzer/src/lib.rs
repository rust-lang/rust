//! Implementation of the LSP for rust-analyzer.
//!
//! This crate takes Rust-specific analysis results from ra_ide and translates
//! into LSP types.
//!
//! It also is the root of all state. `world` module defines the bulk of the
//! state, and `main_loop` module defines the rules for modifying it.
//!
//! The `cli` submodule implements some batch-processing analysis, primarily as
//! a debugging aid.
#![recursion_limit = "512"]

pub mod cli;

#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

mod vfs_glob;
mod caps;
mod cargo_target_spec;
mod to_proto;
mod from_proto;
mod main_loop;
mod markdown;
// TODO: rename to lsp_ext
pub mod req;
pub mod config;
mod world;
mod diagnostics;
mod semantic_tokens;

use serde::de::DeserializeOwned;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub use crate::{
    caps::server_capabilities,
    main_loop::LspError,
    main_loop::{main_loop, show_message},
};

pub fn from_json<T: DeserializeOwned>(what: &'static str, json: serde_json::Value) -> Result<T> {
    let res = T::deserialize(&json)
        .map_err(|e| format!("Failed to deserialize {}: {}; {}", what, e, json))?;
    Ok(res)
}
