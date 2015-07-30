// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(rustc, feature(rustc_private))]
#![cfg_attr(rustdoc, feature(rustdoc))]

#[cfg(rustdoc)]
extern crate rustdoc as this;

#[cfg(rustc)]
extern crate rustc_driver as this;

#[cfg(not(target_os = "nacl"))]
fn main() { this::main() }


// On PNaCl, `libcli_main` defines `main` and in it calls the function
// `nacl_main`, which we define here, by setting the `main_link_name`
// attribute.
#[cfg(target_os = "nacl")] #[main_link_name = "nacl_main"]
fn main() { this::main() }

// All of the following libraries are used so "normal" `main` based programs can
// run as they expect on other platforms.
#[cfg(target_os = "nacl")]

// These libraries communicate with Chrome/the IRT and call into callbacks
// provided by `cli_main`.
#[link(name = "ppapi_cpp", kind = "static")]
#[link(name = "ppapi_simple_cpp", kind = "static")]
#[link(name = "ppapi_stub", kind = "static")]

// `cli_main` wraps the main function, setting up program arguments and the
// program environment in newlib, as one would expect on a non-PPAPI platform.
#[link(name = "cli_main", kind = "static")]

// Required dep of `cli_main`.
#[link(name = "tar", kind = "static")]

// Implements `fork()` and it's ilk by delegating "process"
// creation/waiting/reaping to JS (ie it inserts new embed elements to simulate
// program invocation). It sets up the instance arguments so `cli_main` can do
// its job.
#[link(name = "nacl_spawn", kind = "static")]
extern { }
