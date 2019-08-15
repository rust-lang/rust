#![recursion_limit = "512"]
mod caps;
mod cargo_target_spec;
mod conv;
mod main_loop;
mod markdown;
mod project_model;
pub mod req;
pub mod config;
mod world;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub use crate::{
    caps::server_capabilities, config::ServerConfig, main_loop::main_loop, main_loop::LspError,
};
