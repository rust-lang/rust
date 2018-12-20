mod caps;
mod conv;
mod main_loop;
mod project_model;
pub mod req;
mod server_world;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;
pub use crate::{caps::server_capabilities, main_loop::main_loop, main_loop::LspError};
