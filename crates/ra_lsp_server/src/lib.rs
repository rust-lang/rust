mod caps;
mod cargo_target_spec;
mod conv;
mod main_loop;
mod markdown;
mod project_model;
pub mod req;
pub mod init;
mod server_world;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;
pub use crate::{caps::server_capabilities, main_loop::main_loop, main_loop::LspError, init::InitializationOptions};
