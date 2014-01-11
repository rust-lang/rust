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

use std::{os, run, str, task};
use std::io::process;
pub use std::path::Path;

use extra::workcache;
use rustc::driver::{driver, session};
use rustc::metadata::filesearch;
use rustc::util::sha2;
use syntax::{ast, diagnostic};
use parse_args::{ParseResult, parse_args};
use path_util::{build_pkg_id_in_workspace, default_workspace};
use run_args::run_args;
use context::{BuildContext, Trans, Nothing, Pretty, Analysis, LLVMAssemble};
use context::{LLVMCompileBitcode, BuildCmd, CleanCmd, DoCmd, InfoCmd};
use context::{InstallCmd, ListCmd, PreferCmd, TestCmd, InitCmd, UninstallCmd};
use context::{UnpreferCmd};
use crate_id::CrateId;
use target::{Main};
use workcache_support::digest_only_date;
use exit_codes::{COPY_FAILED_CODE};

pub mod api;
mod conditions;
pub mod context;
mod crate;
pub mod exit_codes;
mod installed_packages;
mod messages;
pub mod crate_id;
pub mod package_source;
mod parse_args;
mod path_util;
mod run_args;
mod source_control;
mod target;
#[cfg(not(windows), test)] // FIXME test failure on windows: #10471
mod tests;
mod util;
pub mod version;
pub mod workcache_support;
mod workspace;

pub mod usage;

/// A PkgScript represents user-supplied custom logic for
/// special build hooks. This only exists for packages with
/// an explicit package script.
struct PkgScript<'a> {
    /// Uniquely identifies this package
    id: &'a CrateId,
    /// File path for the package script
    input: Path,
    /// The session to use *only* for compiling the custom
    /// build script
    sess: session::Session,
    /// The config for compiling the custom build script
    cfg: ast::CrateConfig,
    /// The crate and ast_map for the custom build script
    crate_and_map: Option<(ast::Crate, syntax::ast_map::Map)>,
    /// Directory in which to store build output
    build_dir: Path
}

impl<'a> PkgScript<'a> {
    /// Given the path name for a package script
    /// and a package ID, parse the package script into
    /// a PkgScript that we can then execute
    fn parse<'a>(sysroot: Path,
                 script: Path,
                 workspace: &Path,
                 id: &'a CrateId) -> PkgScript<'a> {
        // Get the executable name that was invoked
        let binary = os::args()[0].to_owned();
        // Build the rustc session data structures to pass
        // to the compiler
        debug!("pkgscript parse: {}", sysroot.display());
        let options = @session::options {
            binary: binary,
            maybe_sysroot: Some(@sysroot),
            outputs: ~[session::OutputExecutable],
            .. (*session::basic_options()).clone()
        };
        let input = driver::file_input(script.clone());
        let sess = driver::build_session(options,
                                         @diagnostic::DefaultEmitter as
                                            @diagnostic::Emitter);
        let cfg = driver::build_configuration(sess);
        let crate = driver::phase_1_parse_input(sess, cfg.clone(), &input);
        let crate_and_map = driver::phase_2_configure_and_expand(sess, cfg.clone(), crate);
        let work_dir = build_pkg_id_in_workspace(id, workspace);

        debug!("Returning package script with id {}", id.to_str());

        PkgScript {
            id: id,
            input: script,
            sess: sess,
            cfg: cfg,
            crate_and_map: Some(crate_and_map),
            build_dir: work_dir
        }
    }

    fn build_custom(&mut self, exec: &mut workcache::Exec) -> ~str {
        let sess = self.sess;

        debug!("Working directory = {}", self.build_dir.display());
        // Collect together any user-defined commands in the package script
        let (crate, ast_map) = self.crate_and_map.take_unwrap();
        let crate = util::ready_crate(sess, crate);
        debug!("Building output filenames with script name {}",
               driver::source_name(&driver::file_input(self.input.clone())));
        let exe = self.build_dir.join("pkg" + util::exe_suffix());
        util::compile_crate_from_input(&self.input,
                                       exec,
                                       Nothing,
                                       &self.build_dir,
                                       sess,
                                       crate,
                                       ast_map,
                                       Main);
        // Discover the output
        // FIXME (#9639): This needs to handle non-utf8 paths
        // Discover the output
        exec.discover_output("binary", exe.as_str().unwrap().to_owned(), digest_only_date(&exe));
        exe.as_str().unwrap().to_owned()
    }


    /// Run the contents of this package script, where <what>
    /// is the command to pass to it (e.g., "build", "clean", "install")
    /// Returns a pair of an exit code and list of configs (obtained by
    /// calling the package script's configs() function if it exists, or
    /// None if `exe` could not be started.
    fn run_custom(exe: &Path, sysroot: &Path) -> Option<(~[~str], process::ProcessExit)> {
        debug!("Running program: {} {} {}", exe.as_str().unwrap().to_owned(),
               sysroot.display(), "install");
        // FIXME #7401 should support commands besides `install`
        // FIXME (#9639): This needs to handle non-utf8 paths
        let opt_status = run::process_status(exe.as_str().unwrap(),
                                             [sysroot.as_str().unwrap().to_owned(), ~"install"]);
        match opt_status {
            Some(status) => {
                if !status.success() {
                    debug!("run_custom: first pkg command failed with {:?}", status);
                    Some((~[], status))
                }
                else {
                    debug!("Running program (configs): {} {} {}",
                           exe.display(), sysroot.display(), "configs");
                    // FIXME (#9639): This needs to handle non-utf8 paths
                    let opt_output = run::process_output(exe.as_str().unwrap(),
                                                         [sysroot.as_str().unwrap().to_owned(),
                                                          ~"configs"]);
                    match opt_output {
                        Some(output) => {
                            debug!("run_custom: second pkg command did {:?}", output.status);
                            // Run the configs() function to get the configs
                            let cfgs = str::from_utf8(output.output).words()
                                .map(|w| w.to_owned()).collect();
                            Some((cfgs, output.status))
                        },
                        None => {
                            debug!("run_custom: second pkg command failed to start");
                            Some((~[], status))
                        }
                    }
                }
            },
            None => {
                debug!("run_custom: first pkg command failed to start");
                None
            }
        }
    }
}

pub fn main() {
    println!("WARNING: The Rust package manager is experimental and may be unstable");
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
        let bctx = BuildContext {
            context: context,
            sysroot: sysroot.clone(), // Currently, only tests override this
            workcache_context: api::default_context(sysroot.clone(),
                                                    default_workspace()).workcache_context
        };
        run_args(&bctx, command, args.clone())
    };
    // FIXME #9262: This is using the same error code for all errors,
    // and at least one test case succeeds if rustpkg returns COPY_FAILED_CODE,
    // when actually, it might set the exit code for that even if a different
    // unhandled condition got raised.
    if result.is_err() { return COPY_FAILED_CODE; }
    return 0;
}
