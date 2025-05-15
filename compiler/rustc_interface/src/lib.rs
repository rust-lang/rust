// tidy-alphabetical-start
#![feature(decl_macro)]
#![feature(file_buffered)]
#![feature(iter_intersperse)]
#![feature(try_blocks)]
// tidy-alphabetical-end

mod callbacks;
pub mod errors;
pub mod interface;
mod limits;
pub mod passes;
mod proc_macro_decls;
mod queries;
pub mod util;

pub use callbacks::setup_callbacks;
pub use interface::{Config, run_compiler};
pub use passes::{DEFAULT_QUERY_PROVIDERS, create_and_enter_global_ctxt, parse};
pub use queries::Linker;

#[cfg(test)]
mod tests;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
