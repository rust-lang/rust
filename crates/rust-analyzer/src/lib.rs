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

mod global_state;
mod main_loop;
mod handlers;
mod caps;
mod cargo_target_spec;
mod to_proto;
mod from_proto;
mod semantic_tokens;
mod markdown;
mod diagnostics;
mod line_endings;
mod request_metrics;
pub mod lsp_ext;
pub mod config;

use serde::de::DeserializeOwned;

pub type Result<T, E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;
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
