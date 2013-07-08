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

#[link(name = "rustpkg",
       vers = "0.8-pre",
       uuid = "25de5e6e-279e-4a20-845c-4cabae92daaf",
       url = "https://github.com/mozilla/rust/tree/master/src/librustpkg")];

#[license = "MIT/ASL2"];
#[crate_type = "lib"];

extern mod extra;
extern mod rustc;
extern mod syntax;

use std::result;
use std::io;
use std::os;
use std::run;
use std::str;

pub use std::path::Path;
use std::hashmap::HashMap;

use rustc::driver::{driver, session};
use rustc::metadata::filesearch;
use extra::{getopts};
use syntax::{ast, diagnostic};
use util::*;
use messages::*;
use path_util::{build_pkg_id_in_workspace, first_pkgid_src_in_workspace};
use path_util::{U_RWX, rust_path};
use path_util::{built_executable_in_workspace, built_library_in_workspace};
use path_util::{target_executable_in_workspace, target_library_in_workspace};
use workspace::{each_pkg_parent_workspace, pkg_parent_workspaces};
use context::Ctx;
use package_id::PkgId;
use package_source::PkgSrc;

pub mod api;
mod conditions;
mod context;
mod crate;
mod messages;
mod package_id;
mod package_path;
mod package_source;
mod path_util;
mod search;
mod target;
#[cfg(test)]
mod tests;
mod util;
mod version;
mod workspace;

pub mod usage;

/// A PkgScript represents user-supplied custom logic for
/// special build hooks. This only exists for packages with
/// an explicit package script.
struct PkgScript<'self> {
    /// Uniquely identifies this package
    id: &'self PkgId,
    // Used to have this field:    deps: ~[(~str, Option<~str>)]
    // but I think it shouldn't be stored here
    /// The contents of the package script: either a file path,
    /// or a string containing the text of the input
    input: driver::input,
    /// The session to use *only* for compiling the custom
    /// build script
    sess: session::Session,
    /// The config for compiling the custom build script
    cfg: ast::crate_cfg,
    /// The crate for the custom build script
    crate: @ast::crate,
    /// Directory in which to store build output
    build_dir: Path
}

impl<'self> PkgScript<'self> {
    /// Given the path name for a package script
    /// and a package ID, parse the package script into
    /// a PkgScript that we can then execute
    fn parse<'a>(script: Path, workspace: &Path, id: &'a PkgId) -> PkgScript<'a> {
        // Get the executable name that was invoked
        let binary = os::args()[0].to_managed();
        // Build the rustc session data structures to pass
        // to the compiler
    debug!("pkgscript parse: %?", os::self_exe_path());
        let options = @session::options {
            binary: binary,
            maybe_sysroot: Some(@os::self_exe_path().get().pop()),
            crate_type: session::bin_crate,
            .. copy *session::basic_options()
        };
        let input = driver::file_input(script);
        let sess = driver::build_session(options, diagnostic::emit);
        let cfg = driver::build_configuration(sess, binary, &input);
        let (crate, _) = driver::compile_upto(sess, copy cfg, &input, driver::cu_parse, None);
        let work_dir = build_pkg_id_in_workspace(id, workspace);

        debug!("Returning package script with id %?", id);

        PkgScript {
            id: id,
            input: input,
            sess: sess,
            cfg: cfg,
            crate: crate.unwrap(),
            build_dir: work_dir
        }
    }

    /// Run the contents of this package script, where <what>
    /// is the command to pass to it (e.g., "build", "clean", "install")
    /// Returns a pair of an exit code and list of configs (obtained by
    /// calling the package script's configs() function if it exists
    // FIXME (#4432): Use workcache to only compile the script when changed
    fn run_custom(&self, sysroot: @Path) -> (~[~str], ExitCode) {
        let sess = self.sess;

        debug!("Working directory = %s", self.build_dir.to_str());
        // Collect together any user-defined commands in the package script
        let crate = util::ready_crate(sess, self.crate);
        debug!("Building output filenames with script name %s",
               driver::source_name(&self.input));
        match filesearch::get_rustpkg_sysroot() {
            Ok(r) => {
                let root = r.pop().pop().pop().pop(); // :-\
                debug!("Root is %s, calling compile_rest", root.to_str());
                let exe = self.build_dir.push(~"pkg" + util::exe_suffix());
                let binary = os::args()[0].to_managed();
                util::compile_crate_from_input(&self.input,
                                               &self.build_dir,
                                               sess,
                                               crate,
                                               driver::build_configuration(sess,
                                                                           binary, &self.input),
                                               driver::cu_parse);
                debug!("Running program: %s %s %s %s", exe.to_str(),
                       sysroot.to_str(), root.to_str(), "install");
                // FIXME #7401 should support commands besides `install`
                let status = run::process_status(exe.to_str(), [sysroot.to_str(), ~"install"]);
                if status != 0 {
                    return (~[], status);
                }
                else {
                    debug!("Running program (configs): %s %s %s",
                           exe.to_str(), root.to_str(), "configs");
                    let output = run::process_output(exe.to_str(), [root.to_str(), ~"configs"]);
                    // Run the configs() function to get the configs
                    let cfgs = str::from_bytes_slice(output.output).word_iter()
                        .transform(|w| w.to_owned()).collect();
                    (cfgs, output.status)
                }
            }
            Err(e) => {
                fail!("Running package script, couldn't find rustpkg sysroot (%s)", e)
            }
        }
    }

    fn hash(&self) -> ~str {
        self.id.hash()
    }

}

pub trait CtxMethods {
    fn run(&self, cmd: &str, args: ~[~str]);
    fn do_cmd(&self, _cmd: &str, _pkgname: &str);
    fn build(&self, workspace: &Path, pkgid: &PkgId);
    fn clean(&self, workspace: &Path, id: &PkgId);
    fn info(&self);
    fn install(&self, workspace: &Path, id: &PkgId);
    fn install_no_build(&self, workspace: &Path, id: &PkgId);
    fn prefer(&self, _id: &str, _vers: Option<~str>);
    fn test(&self);
    fn uninstall(&self, _id: &str, _vers: Option<~str>);
    fn unprefer(&self, _id: &str, _vers: Option<~str>);
}

impl CtxMethods for Ctx {

    fn run(&self, cmd: &str, args: ~[~str]) {
        match cmd {
            "build" => {
                if args.len() < 1 {
                    return usage::build();
                }
                // The package id is presumed to be the first command-line
                // argument
                let pkgid = PkgId::new(copy args[0]);
                for each_pkg_parent_workspace(&pkgid) |workspace| {
                    self.build(workspace, &pkgid);
                }
            }
            "clean" => {
                if args.len() < 1 {
                    return usage::build();
                }
                // The package id is presumed to be the first command-line
                // argument
                let pkgid = PkgId::new(copy args[0]);
                let cwd = os::getcwd();
                self.clean(&cwd, &pkgid); // tjc: should use workspace, not cwd
            }
            "do" => {
                if args.len() < 2 {
                    return usage::do_cmd();
                }

                self.do_cmd(copy args[0], copy args[1]);
            }
            "info" => {
                self.info();
            }
            "install" => {
                if args.len() < 1 {
                    return usage::install();
                }

                // The package id is presumed to be the first command-line
                // argument
                let pkgid = PkgId::new(args[0]);
                let workspaces = pkg_parent_workspaces(&pkgid);
                if workspaces.is_empty() {
                    let rp = rust_path();
                    assert!(!rp.is_empty());
                    let src = PkgSrc::new(&rp[0], &build_pkg_id_in_workspace(&pkgid, &rp[0]),
                                          &pkgid);
                    src.fetch_git();
                    self.install(&rp[0], &pkgid);
                }
                else {
                    for each_pkg_parent_workspace(&pkgid) |workspace| {
                        self.install(workspace, &pkgid);
                    }
                }
            }
            "prefer" => {
                if args.len() < 1 {
                    return usage::uninstall();
                }

                self.prefer(args[0], None);
            }
            "test" => {
                self.test();
            }
            "uninstall" => {
                if args.len() < 1 {
                    return usage::uninstall();
                }

                self.uninstall(args[0], None);
            }
            "unprefer" => {
                if args.len() < 1 {
                    return usage::uninstall();
                }

                self.unprefer(args[0], None);
            }
            _ => fail!(fmt!("I don't know the command `%s`", cmd))
        }
    }

    fn do_cmd(&self, _cmd: &str, _pkgname: &str)  {
        // stub
        fail!("`do` not yet implemented");
    }

    fn build(&self, workspace: &Path, pkgid: &PkgId) {
        debug!("build: workspace = %s pkgid = %s", workspace.to_str(),
               pkgid.to_str());
        let src_dir   = first_pkgid_src_in_workspace(pkgid, workspace);
        let build_dir = build_pkg_id_in_workspace(pkgid, workspace);
        debug!("Destination dir = %s", build_dir.to_str());

        // Create the package source
        let mut src = PkgSrc::new(workspace, &build_dir, pkgid);
        debug!("Package src = %?", src);

        // Is there custom build logic? If so, use it
        let pkg_src_dir = src_dir;
        let mut custom = false;
        debug!("Package source directory = %?", pkg_src_dir);
        let cfgs = match pkg_src_dir.chain_ref(|p| src.package_script_option(p)) {
            Some(package_script_path) => {
                let pscript = PkgScript::parse(package_script_path,
                                               workspace,
                                               pkgid);
                let sysroot = self.sysroot_opt.expect("custom build needs a sysroot");
                let (cfgs, hook_result) = pscript.run_custom(sysroot);
                debug!("Command return code = %?", hook_result);
                if hook_result != 0 {
                    fail!("Error running custom build command")
                }
                custom = true;
                // otherwise, the package script succeeded
                cfgs
            }
            None => {
                debug!("No package script, continuing");
                ~[]
            }
        };

        // If there was a package script, it should have finished
        // the build already. Otherwise...
        if !custom {
            // Find crates inside the workspace
            src.find_crates();
            // Build it!
            src.build(self, build_dir, cfgs);
        }
    }

    fn clean(&self, workspace: &Path, id: &PkgId)  {
        // Could also support a custom build hook in the pkg
        // script for cleaning files rustpkg doesn't know about.
        // Do something reasonable for now

        let dir = build_pkg_id_in_workspace(id, workspace);
        note(fmt!("Cleaning package %s (removing directory %s)",
                        id.to_str(), dir.to_str()));
        if os::path_exists(&dir) {
            os::remove_dir_recursive(&dir);
            note(fmt!("Removed directory %s", dir.to_str()));
        }

        note(fmt!("Cleaned package %s", id.to_str()));
    }

    fn info(&self) {
        // stub
        fail!("info not yet implemented");
    }

    fn install(&self, workspace: &Path, id: &PkgId)  {
        // FIXME #7402: Use RUST_PATH to determine target dir
        // Also should use workcache to not build if not necessary.
        self.build(workspace, id);
        debug!("install: workspace = %s, id = %s", workspace.to_str(),
               id.to_str());
        self.install_no_build(workspace, id);

    }

    fn install_no_build(&self, workspace: &Path, id: &PkgId) {
        use conditions::copy_failed::cond;

        // Now copy stuff into the install dirs
        let maybe_executable = built_executable_in_workspace(id, workspace);
        let maybe_library = built_library_in_workspace(id, workspace);
        let target_exec = target_executable_in_workspace(id, workspace);
        let target_lib = maybe_library.map(|_p| target_library_in_workspace(id, workspace));

        debug!("target_exec = %s target_lib = %? \
                maybe_executable = %? maybe_library = %?",
               target_exec.to_str(), target_lib,
               maybe_executable, maybe_library);

        for maybe_executable.iter().advance |exec| {
            debug!("Copying: %s -> %s", exec.to_str(), target_exec.to_str());
            if !(os::mkdir_recursive(&target_exec.dir_path(), U_RWX) &&
                 os::copy_file(exec, &target_exec)) {
                cond.raise((copy *exec, copy target_exec));
            }
        }
        for maybe_library.iter().advance |lib| {
            let target_lib = (copy target_lib).expect(fmt!("I built %s but apparently \
                                                didn't install it!", lib.to_str()));
            debug!("Copying: %s -> %s", lib.to_str(), target_lib.to_str());
            if !(os::mkdir_recursive(&target_lib.dir_path(), U_RWX) &&
                 os::copy_file(lib, &target_lib)) {
                cond.raise((copy *lib, copy target_lib));
            }
        }
    }

    fn prefer(&self, _id: &str, _vers: Option<~str>)  {
        fail!("prefer not yet implemented");
    }

    fn test(&self)  {
        // stub
        fail!("test not yet implemented");
    }

    fn uninstall(&self, _id: &str, _vers: Option<~str>)  {
        fail!("uninstall not yet implemented");
    }

    fn unprefer(&self, _id: &str, _vers: Option<~str>)  {
        fail!("unprefer not yet implemented");
    }
}


pub fn main() {
    io::println("WARNING: The Rust package manager is experimental and may be unstable");

    let args = os::args();
    let opts = ~[getopts::optflag("h"), getopts::optflag("help"),
                 getopts::optflag("j"), getopts::optflag("json"),
                 getopts::optmulti("c"), getopts::optmulti("cfg")];
    let matches = &match getopts::getopts(args, opts) {
        result::Ok(m) => m,
        result::Err(f) => {
            error(fmt!("%s", getopts::fail_str(f)));

            return;
        }
    };
    let help = getopts::opt_present(matches, "h") ||
               getopts::opt_present(matches, "help");
    let json = getopts::opt_present(matches, "j") ||
               getopts::opt_present(matches, "json");
    let mut args = copy matches.free;

    args.shift();

    if (args.len() < 1) {
        return usage::general();
    }

    let cmd = args.shift();

    if !util::is_cmd(cmd) {
        return usage::general();
    } else if help {
        return match cmd {
            ~"build" => usage::build(),
            ~"clean" => usage::clean(),
            ~"do" => usage::do_cmd(),
            ~"info" => usage::info(),
            ~"install" => usage::install(),
            ~"prefer" => usage::prefer(),
            ~"test" => usage::test(),
            ~"uninstall" => usage::uninstall(),
            ~"unprefer" => usage::unprefer(),
            _ => usage::general()
        };
    }

    let sroot = match filesearch::get_rustpkg_sysroot() {
        Ok(r) => Some(@r.pop().pop()), Err(_) => None
    };
    debug!("Using sysroot: %?", sroot);
    Ctx {
        sysroot_opt: sroot, // Currently, only tests override this
        json: json,
        dep_cache: @mut HashMap::new()
    }.run(cmd, args);
}

/**
 * Get the working directory of the package script.
 * Assumes that the package script has been compiled
 * in is the working directory.
 */
pub fn work_dir() -> Path {
    os::self_exe_path().get()
}

/**
 * Get the source directory of the package (i.e.
 * where the crates are located). Assumes
 * that the cwd is changed to it before
 * running this executable.
 */
pub fn src_dir() -> Path {
    os::getcwd()
}
