// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustpkg - a package manager and build system for Rust

#[crate_id = "rustpkg#0.9"];
#[license = "MIT/ASL2"];
#[crate_type = "dylib"];

#[feature(globs, managed_boxes)];

extern mod extra;
extern mod rustc;
extern mod syntax;

use std::path::Path;
use std::{os, task};

use context::{Trans, Nothing, Pretty, Analysis, LLVMAssemble};
use context::{LLVMCompileBitcode, BuildCmd, CleanCmd, DoCmd, InfoCmd};
use context::{InstallCmd, ListCmd, PreferCmd, TestCmd, InitCmd, UninstallCmd};
use context::{UnpreferCmd};
use context::{BuildContext};
use rustc::metadata::filesearch;
use run_cmd::{run_cmd};
use parse_args::{ParseResult, parse_args};
use path_util::{default_workspace};
use exit_codes::{COPY_FAILED_CODE};

pub mod api;
mod conditions;
pub mod context;
mod crate;
pub mod exit_codes;
mod installed_packages;
mod messages;
pub mod crate_id;
mod package_script;
pub mod package_source;
mod parse_args;
mod path_util;
mod run_cmd;
mod source_control;
mod target;
#[cfg(not(windows), test)] // FIXME test failure on windows: #10471
mod tests;
pub mod usage;
mod util;
pub mod version;
pub mod workcache_support;
mod workspace;


pub fn main() {
    println("WARNING: The Rust package manager is experimental and may be unstable");
    os::set_exit_status(main_args(os::args()));
}

pub fn main_args(args: &[~str]) -> int {

    let (command, args, context, supplied_sysroot) = match parse_args(args) {
        Ok(ParseResult {
            command: cmd,
            args: args,
            context: ctx,
            sysroot: sroot}) => (cmd, args, ctx, sroot),
        Err(error_code) => {
            debug!("Parsing failed. Returning error code {}", error_code);
            return error_code
        }
    };
    debug!("Finished parsing commandline args {:?}", args);
    debug!("  Using command: {:?}", command);
    debug!("  Using args {:?}", args);
    debug!("  Using cflags: {:?}", context.rustc_flags);
    debug!("  Using rust_path_hack {:b}", context.use_rust_path_hack);
    debug!("  Using cfgs: {:?}", context.cfgs);
    debug!("  Using supplied_sysroot: {:?}", supplied_sysroot);

    let sysroot = match supplied_sysroot {
        Some(s) => Path::new(s),
        _ => filesearch::get_or_default_sysroot()
    };

    debug!("Using sysroot: {}", sysroot.display());
    let ws = default_workspace();
    debug!("Will store workcache in {}", ws.display());

    // Wrap the rest in task::try in case of a condition failure in a task
    let result = do task::try {
        let build_context = BuildContext {
            context: context,
            sysroot: sysroot.clone(), // Currently, only tests override this
            workcache_context: api::default_context(sysroot.clone(),
                                                    default_workspace()).workcache_context
        };
        run_cmd(command, args.clone(), &build_context);
    };
    // FIXME #9262: This is using the same error code for all errors,
    // and at least one test case succeeds if rustpkg returns COPY_FAILED_CODE,
    // when actually, it might set the exit code for that even if a different
    // unhandled condition got raised.
    if result.is_err() { return COPY_FAILED_CODE; }
    return 0;
}
