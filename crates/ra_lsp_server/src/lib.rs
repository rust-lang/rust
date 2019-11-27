//! Implementation of the LSP for rust-analyzer.
//!
//! This crate takes Rust-specific analysis results from ra_ide and
//! translates into LSP types.
//!
//! It also is the root of all state. `world` module defines the bulk of the
//! state, and `main_loop` module defines the rules for modifying it.
#![recursion_limit = "512"]

#[allow(unused)]
macro_rules! println {
    ($($tt:tt)*) => {
        compile_error!("stdout is locked, use eprintln")
    };
}

#[allow(unused)]
macro_rules! print {
    ($($tt:tt)*) => {
        compile_error!("stdout is locked, use eprint")
    };
}

mod caps;
mod cargo_target_spec;
mod conv;
mod main_loop;
mod markdown;
pub mod req;
mod config;
mod world;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub use crate::{
    caps::server_capabilities,
    config::ServerConfig,
    main_loop::LspError,
    main_loop::{main_loop, show_message},
};
