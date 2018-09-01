#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate languageserver_types;
#[macro_use]
extern crate crossbeam_channel;
extern crate threadpool;
#[macro_use]
extern crate log;
extern crate drop_bomb;
extern crate url_serde;
extern crate walkdir;
extern crate libeditor;
extern crate libanalysis;
extern crate libsyntax2;
extern crate gen_lsp_server;
extern crate im;
extern crate relative_path;

mod caps;
pub mod req;
mod conv;
mod main_loop;
mod vfs;
mod path_map;
mod server_world;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;
pub use caps::server_capabilities;
pub use main_loop::main_loop;

