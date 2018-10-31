#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;
extern crate languageserver_types;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate crossbeam_channel;
extern crate rayon;
#[macro_use]
extern crate log;
extern crate cargo_metadata;
extern crate drop_bomb;
#[macro_use]
extern crate failure_derive;
extern crate im;
extern crate relative_path;
extern crate rustc_hash;
extern crate url_serde;
extern crate walkdir;

extern crate gen_lsp_server;
extern crate ra_analysis;
extern crate ra_editor;
extern crate ra_syntax;

mod caps;
mod conv;
mod main_loop;
mod path_map;
mod project_model;
pub mod req;
mod server_world;
pub mod thread_watcher;
mod vfs;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;
pub use crate::{caps::server_capabilities, main_loop::main_loop, main_loop::LspError};
