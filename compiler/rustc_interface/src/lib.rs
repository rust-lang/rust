// tidy-alphabetical-start
#![feature(decl_macro)]
#![feature(file_buffered)]
#![feature(iter_intersperse)]
#![feature(let_chains)]
#![feature(try_blocks)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

mod callbacks;
mod errors;
pub mod interface;
pub mod passes;
mod proc_macro_decls;
mod queries;
pub mod util;

pub use callbacks::setup_callbacks;
pub use interface::{Config, run_compiler};
pub use passes::DEFAULT_QUERY_PROVIDERS;
pub use queries::{Linker, Queries};

#[cfg(test)]
mod tests;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
