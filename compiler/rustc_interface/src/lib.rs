#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(internal_output_capture)]
#![feature(thread_spawn_unchecked)]
#![feature(nll)]
#![feature(once_cell)]
#![recursion_limit = "256"]

mod callbacks;
pub mod interface;
mod passes;
mod proc_macro_decls;
mod queries;
pub mod util;

pub use interface::{run_compiler, Config};
pub use passes::{DEFAULT_EXTERN_QUERY_PROVIDERS, DEFAULT_QUERY_PROVIDERS};
pub use queries::Queries;

#[cfg(test)]
mod tests;
