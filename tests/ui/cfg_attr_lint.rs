#![feature(tool_lints)]

#![warn(clippy::deprecated_cfg_attr)]

// This doesn't get linted, see known problems
#![cfg_attr(rustfmt, rustfmt_skip)]

#[cfg_attr(rustfmt, rustfmt_skip)]
fn main() {}
