#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(let_else)]
#![feature(internal_output_capture)]
#![feature(thread_spawn_unchecked)]
#![feature(nll)]
#![feature(once_cell)]
#![recursion_limit = "256"]
#![cfg_attr(not(bootstrap), allow(rustc::potential_query_instability))]

mod callbacks;
pub mod interface;
mod passes;
mod proc_macro_decls;
mod queries;
pub mod util;

pub use callbacks::setup_callbacks;
pub use interface::{run_compiler, Config};
pub use passes::{DEFAULT_EXTERN_QUERY_PROVIDERS, DEFAULT_QUERY_PROVIDERS};
pub use queries::Queries;

#[cfg(test)]
mod tests;
