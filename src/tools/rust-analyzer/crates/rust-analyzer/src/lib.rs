//! Implementation of the LSP for rust-analyzer.
//!
//! This crate takes Rust-specific analysis results from ide and translates
//! into LSP types.
//!
//! It also is the root of all state. `world` module defines the bulk of the
//! state, and `main_loop` module defines the rules for modifying it.
//!
//! The `cli` submodule implements some batch-processing analysis, primarily as
//! a debugging aid.

#![warn(rust_2018_idioms, unused_lifetimes)]

pub mod cli;

mod caps;
mod cargo_target_spec;
mod diagnostics;
mod diff;
mod dispatch;
mod global_state;
mod hack_recover_crate_name;
mod line_index;
mod main_loop;
mod mem_docs;
mod op_queue;
mod reload;
mod task_pool;
mod version;

mod handlers {
    pub(crate) mod notification;
    pub(crate) mod request;
}

pub mod tracing {
    pub mod config;
    pub use config::Config;
    pub mod hprof;
}

pub mod config;
pub mod lsp;
use self::lsp::ext as lsp_ext;

#[cfg(test)]
mod integrated_benchmarks;

use serde::de::DeserializeOwned;

pub use crate::{
    caps::server_capabilities, main_loop::main_loop, reload::ws_to_crate_graph, version::version,
};

pub fn from_json<T: DeserializeOwned>(
    what: &'static str,
    json: &serde_json::Value,
) -> anyhow::Result<T> {
    serde_json::from_value(json.clone())
        .map_err(|e| anyhow::format_err!("Failed to deserialize {what}: {e}; {json}"))
}
