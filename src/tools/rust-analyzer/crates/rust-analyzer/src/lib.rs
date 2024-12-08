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

pub mod cli;

mod command;
mod diagnostics;
mod discover;
mod flycheck;
mod hack_recover_crate_name;
mod line_index;
mod main_loop;
mod mem_docs;
mod op_queue;
mod reload;
mod target_spec;
mod task_pool;
mod test_runner;
mod version;

mod handlers {
    pub(crate) mod dispatch;
    pub(crate) mod notification;
    pub(crate) mod request;
}

pub mod tracing {
    pub mod config;
    pub mod json;
    pub use config::Config;
    pub mod hprof;
}

pub mod config;
mod global_state;
pub mod lsp;
use self::lsp::ext as lsp_ext;

#[cfg(test)]
mod integrated_benchmarks;

use serde::de::DeserializeOwned;

pub use crate::{
    lsp::capabilities::server_capabilities, main_loop::main_loop, reload::ws_to_crate_graph,
    version::version,
};

pub fn from_json<T: DeserializeOwned>(
    what: &'static str,
    json: &serde_json::Value,
) -> anyhow::Result<T> {
    serde_json::from_value(json.clone())
        .map_err(|e| anyhow::format_err!("Failed to deserialize {what}: {e}; {json}"))
}
