//! rustbuild, the Rust build system
//!
//! This is the entry point for the build system used to compile the `rustc`
//! compiler. Lots of documentation can be found in the `README.md` file in the
//! parent directory, and otherwise documentation can be found throughout the `build`
//! directory in each respective module.

#![deny(warnings)]

use std::env;

use bootstrap::{Config, Build};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let config = Config::parse(&args);
    Build::new(config).build();
}
