// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private)]
#![feature(link_args)]

// Set the stack size at link time on Windows. See rustc_driver::in_rustc_thread
// for the rationale.
#[allow(unused_attributes)]
#[cfg_attr(all(windows, target_env = "msvc"), link_args = "/STACK:16777216")]
// We only build for msvc and gnu now, but we use a exhaustive condition here
// so we can expect either the stack size to be set or the build fails.
#[cfg_attr(all(windows, not(target_env = "msvc")), link_args = "-Wl,--stack,16777216")]
// Also, don't forget to set this for rustdoc.
extern {}

extern crate rustc_driver;

fn main() {
    rustc_driver::set_sigpipe_handler();
    rustc_driver::main()
}
