// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! General comments on the build system:
//!
//! The Rust Build System is an alternative build system for building
//! the Rust compiler. It is written in Rust and is managed as a
//! Cargo package, `build-rust`. The user will initiate the build
//! process by running `cargo run`, under its manifest directory.
//!
//! The build process starts with Cargo invoking a small build script,
//! `build.rs`, located at the same directory as the package manifest.
//! This script will collect the triple of the build machine and
//! the path to the manifest directory, and then generate a file
//! `build_env.rs`, which will be included into `configure.rs`.
//!
//! The next stage of the build process starts in module `configure`.
//! The build system will invoke `configure::configure()`, which will
//! parse the command line arguments, inspect the build environment
//! (for instance, check the availability of the required build
//! programs), and then return a ConfigArgs object which encapsulates
//! the information collected. Future build processes will read this
//! object instead of poking the build environment directly.
//!
//! Because the configure step may fail (for instance, it may be
//! unable to find the required build program), the `configure()`
//! returns type `BuildState<T>` where `T` equals `ConfigArgs`.
//! `BuildState<T>` is a wrapper around the `Result<T, E>` which is
//! used to indicate the success/failure state of a function.
//! For details, see `mod build_state`.
//!
//! The build system will then download the stage0 snapshot,
//! configure and build LLVM, invoke the appropriate toolchain to
//! build runtime libraries, and then finally boostrap a working stage2
//! rustc. For details of these steps, see the respective modules for
//! more comments.

extern crate regex;

#[macro_use]
mod build_state;
mod cmd_args;
mod configure;
mod snapshot;
mod llvm;
mod rt;
mod rust;
mod cc;
mod log;

use build_state::*;
use configure::{ConfigArgs, configure};
use llvm::{build_llvm, configure_llvm};
use rt::build_native_libs;
use rust::build_rust;
use snapshot::download_stage0_snapshot;

fn make(args : &ConfigArgs) -> BuildState<()> {
    let dl = download_stage0_snapshot(args);
    if !args.no_reconfigure_llvm() {
        try!(configure_llvm(args));
    }
    if !args.no_rebuild_llvm() {
        try!(build_llvm(args));
    }
    if !args.no_bootstrap() {
        try!(build_native_libs(args));
    }
    try!(dl.recv().unwrap());   // we need to wait for stage0 download
    try!(build_rust(args));
    continue_build()
}

fn run() -> BuildState<()> {
    let config_args = try!(configure());

    try!(make(&config_args));

    success_stop()
}

fn main() {
    let result : BuildState<()> = run();
    match result {
        Err(ExitStatus::MsgStop(e)) => println!("{}", e),
        Err(ExitStatus::ErrStop(e)) => println!("Build failed: {}", e),
        _ => println!("Build successful."),
    }
}
