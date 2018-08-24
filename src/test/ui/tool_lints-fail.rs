// Don't allow tool_lints, which aren't scoped

#![feature(tool_lints)]
#![deny(unknown_lints)]

#![deny(clippy)] //~ ERROR: unknown lint: `clippy`

fn main() {}
