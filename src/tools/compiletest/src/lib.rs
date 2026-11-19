#![crate_name = "compiletest"]
#![warn(unreachable_pub)]

#[cfg(test)]
mod tests;

// Public modules needed by the compiletest binary or by `rustdoc-gui-test`.
pub mod cli;
pub mod rustdoc_gui_test;

mod common;
mod debuggers;
mod diagnostics;
mod directives;
mod discovery;
mod edition;
mod errors;
mod executor;
mod json;
mod output_capture;
mod panic_hook;
mod raise_fd_limit;
mod read2;
mod runtest;
mod util;
