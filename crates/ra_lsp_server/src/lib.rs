mod caps;
mod cargo_target_spec;
mod conv;
mod main_loop;
mod markdown;
mod project_model;
mod vfs_filter;
pub mod req;
pub mod init;
mod world;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub use crate::{caps::server_capabilities, main_loop::main_loop, main_loop::LspError, init::InitializationOptions};
