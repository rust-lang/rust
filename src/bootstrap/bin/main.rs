// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! rustbuild, the Rust build system
//!
//! This is the entry point for the build system used to compile the `rustc`
//! compiler. Lots of documentation can be found in the `README.md` file next to
//! this file, and otherwise documentation can be found throughout the `build`
//! directory in each respective module.

#![deny(warnings)]

extern crate bootstrap;

use std::env;

use bootstrap::{Flags, Config, Build};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let flags = Flags::parse(&args);
    let mut config = Config::parse(&flags.build, flags.config.clone());

    // compat with `./configure` while we're still using that
    if std::fs::metadata("config.mk").is_ok() {
        config.update_with_config_mk();
    }

    Build::new(flags, config).build();
}
