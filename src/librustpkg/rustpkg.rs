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
       vers = "0.9-pre",
       uuid = "25de5e6e-279e-4a20-845c-4cabae92daaf",
       url = "https://github.com/mozilla/rust/tree/master/src/librustpkg")];

#[license = "MIT/ASL2"];
#[crate_type = "lib"];

extern mod extra;
extern mod rustc;
extern mod syntax;

use std::{io, os, result, run, str, task};
pub use std::path::Path;

use extra::workcache;
use rustc::driver::{driver, session};
use rustc::metadata::filesearch;
use rustc::metadata::filesearch::rust_path;
use extra::{getopts};
use syntax::{ast, diagnostic};
use util::*;
use messages::{error, warn, note};
use path_util::{build_pkg_id_in_workspace, built_test_in_workspace};
use path_util::{U_RWX, in_rust_path};
use path_util::{built_executable_in_workspace, built_library_in_workspace, default_workspace};
use path_util::{target_executable_in_workspace, target_library_in_workspace};
use source_control::is_git_dir;
use workspace::{each_pkg_parent_workspace, pkg_parent_workspaces, cwd_to_workspace};
use context::{Context, BuildContext,
                       RustcFlags, Trans, Link, Nothing, Pretty, Analysis, Assemble,
                       LLVMAssemble, LLVMCompileBitcode};
use package_id::PkgId;
use package_source::PkgSrc;
use target::{WhatToBuild, Everything, is_lib, is_main, is_test, is_bench, Tests};
// use workcache_support::{discover_outputs, digest_only_date};
use workcache_support::digest_only_date;
use exit_codes::COPY_FAILED_CODE;

pub mod api;
mod conditions;
mod context;
mod crate;
mod exit_codes;
mod installed_packages;
mod messages;
mod package_id;
mod package_source;
mod path_util;
mod search;
mod source_control;
mod target;
#[cfg(test)]
mod tests;
mod util;
mod version;
pub mod workcache_support;
mod workspace;

pub mod usage;

/// A PkgScript represents user-supplied custom logic for
/// special build hooks. This only exists for packages with
/// an explicit package script.
struct PkgScript<'self> {
    /// Uniquely identifies this package
    id: &'self PkgId,
    /// File path for the package script
    input: Path,
    /// The session to use *only* for compiling the custom
    /// build script
    sess: session::Session,
    /// The config for compiling the custom build script
    cfg: ast::CrateConfig,
    /// The crate for the custom build script
    crate: Option<ast::Crate>,
    /// Directory in which to store build output
    build_dir: Path
}

impl<'self> PkgScript<'self> {
    /// Given the path name for a package script
    /// and a package ID, parse the package script into
    /// a PkgScript that we can then execute
    fn parse<'a>(sysroot: @Path,
                 script: Path,
                 workspace: &Path,
                 id: &'a PkgId) -> PkgScript<'a> {
        // Get the executable name that was invoked
        let binary = os::args()[0].to_managed();
        // Build the rustc session data structures to pass
        // to the compiler
        debug2!("pkgscript parse: {}", sysroot.to_str());
        let options = @session::options {
            binary: binary,
            maybe_sysroot: Some(sysroot),
            crate_type: session::bin_crate,
            .. (*session::basic_options()).clone()
        };
        let input = driver::file_input(script.clone());
        let sess = driver::build_session(options,
                                         @diagnostic::DefaultEmitter as
                                            @diagnostic::Emitter);
        let cfg = driver::build_configuration(sess);
        let crate = driver::phase_1_parse_input(sess, cfg.clone(), &input);
        let crate = driver::phase_2_configure_and_expand(sess, cfg.clone(), crate);
        let work_dir = build_pkg_id_in_workspace(id, workspace);

        debug2!("Returning package script with id {}", id.to_str());

        PkgScript {
            id: id,
            input: script,
            sess: sess,
            cfg: cfg,
            crate: Some(crate),
            build_dir: work_dir
        }
    }

    /// Run the contents of this package script, where <what>
    /// is the command to pass to it (e.g., "build", "clean", "install")
    /// Returns a pair of an exit code and list of configs (obtained by
    /// calling the package script's configs() function if it exists
    fn run_custom(&mut self, exec: &mut workcache::Exec,
                  sysroot: &Path) -> (~[~str], ExitCode) {
        let sess = self.sess;

        debug2!("Working directory = {}", self.build_dir.to_str());
        // Collect together any user-defined commands in the package script
        let crate = util::ready_crate(sess, self.crate.take_unwrap());
        debug2!("Building output filenames with script name {}",
               driver::source_name(&driver::file_input(self.input.clone())));
        let exe = self.build_dir.push(~"pkg" + util::exe_suffix());
        util::compile_crate_from_input(&self.input,
                                       exec,
                                       Nothing,
                                       &self.build_dir,
                                       sess,
                                       crate);
        debug2!("Running program: {} {} {}", exe.to_str(),
               sysroot.to_str(), "install");
        // Discover the output
        exec.discover_output("binary", exe.to_str(), digest_only_date(&exe));
        // FIXME #7401 should support commands besides `install`
        let status = run::process_status(exe.to_str(), [sysroot.to_str(), ~"install"]);
        if status != 0 {
            return (~[], status);
        }
        else {
            debug2!("Running program (configs): {} {} {}",
                   exe.to_str(), sysroot.to_str(), "configs");
            let output = run::process_output(exe.to_str(), [sysroot.to_str(), ~"configs"]);
            // Run the configs() function to get the configs
            let cfgs = str::from_utf8_slice(output.output).word_iter()
                .map(|w| w.to_owned()).collect();
            (cfgs, output.status)
        }
    }

    fn hash(&self) -> ~str {
        self.id.hash()
    }
}

pub trait CtxMethods {
    fn run(&self, cmd: &str, args: ~[~str]);
    fn do_cmd(&self, _cmd: &str, _pkgname: &str);
    /// Returns a pair of the selected package ID, and the destination workspace
    fn build_args(&self, args: ~[~str], what: &WhatToBuild) -> Option<(PkgId, Path)>;
    /// Returns the destination workspace
    fn build(&self, pkg_src: &mut PkgSrc, what: &WhatToBuild) -> Path;
    fn clean(&self, workspace: &Path, id: &PkgId);
    fn info(&self);
    /// Returns a pair. First component is a list of installed paths,
    /// second is a list of declared and discovered inputs
    fn install(&self, src: PkgSrc, what: &WhatToBuild) -> (~[Path], ~[(~str, ~str)]);
    /// Returns a list of installed files
    fn install_no_build(&self,
                        source_workspace: &Path,
                        target_workspace: &Path,
                        id: &PkgId) -> ~[~str];
    fn prefer(&self, _id: &str, _vers: Option<~str>);
    fn test(&self, id: &PkgId, workspace: &Path);
    fn uninstall(&self, _id: &str, _vers: Option<~str>);
    fn unprefer(&self, _id: &str, _vers: Option<~str>);
    fn init(&self);
}

impl CtxMethods for BuildContext {
    fn build_args(&self, args: ~[~str], what: &WhatToBuild) -> Option<(PkgId, Path)> {
        if args.len() < 1 {
            match cwd_to_workspace() {
                None if self.context.use_rust_path_hack => {
                    let cwd = os::getcwd();
                    let pkgid = PkgId::new(cwd.components[cwd.components.len() - 1]);
                    let mut pkg_src = PkgSrc::new(cwd, true, pkgid);
                    let dest_ws = self.build(&mut pkg_src, what);
                    Some((pkg_src.id, dest_ws))
                }
                None => { usage::build(); None }
                Some((ws, pkgid)) => {
                    let mut pkg_src = PkgSrc::new(ws, false, pkgid);
                    let dest_ws = self.build(&mut pkg_src, what);
                    Some((pkg_src.id, dest_ws))
                }
            }
        } else {
            // The package id is presumed to be the first command-line
            // argument
            let pkgid = PkgId::new(args[0].clone());
            let mut dest_ws = None;
            do each_pkg_parent_workspace(&self.context, &pkgid) |workspace| {
                debug2!("found pkg {} in workspace {}, trying to build",
                       pkgid.to_str(), workspace.to_str());
                let mut pkg_src = PkgSrc::new(workspace.clone(), false, pkgid.clone());
                dest_ws = Some(self.build(&mut pkg_src, what));
                true
            };
            assert!(dest_ws.is_some());
            // n.b. If this builds multiple packages, it only returns the workspace for
            // the last one. The whole building-multiple-packages-with-the-same-ID is weird
            // anyway and there are no tests for it, so maybe take it out
            Some((pkgid, dest_ws.unwrap()))
        }
    }
    fn run(&self, cmd: &str, args: ~[~str]) {
        match cmd {
            "build" => {
                self.build_args(args, &Everything);
            }
            "clean" => {
                if args.len() < 1 {
                    match cwd_to_workspace() {
                        None => { usage::clean(); return }
                        // tjc: Maybe clean should clean all the packages in the
                        // current workspace, though?
                        Some((ws, pkgid)) => self.clean(&ws, &pkgid)
                    }

                }
                else {
                    // The package id is presumed to be the first command-line
                    // argument
                    let pkgid = PkgId::new(args[0].clone());
                    let cwd = os::getcwd();
                    self.clean(&cwd, &pkgid); // tjc: should use workspace, not cwd
                }
            }
            "do" => {
                if args.len() < 2 {
                    return usage::do_cmd();
                }

                self.do_cmd(args[0].clone(), args[1].clone());
            }
            "info" => {
                self.info();
            }
            "install" => {
               if args.len() < 1 {
                    match cwd_to_workspace() {
                        None if self.context.use_rust_path_hack => {
                            let cwd = os::getcwd();
                            let inferred_pkgid =
                                PkgId::new(cwd.components[cwd.components.len() - 1]);
                            self.install(PkgSrc::new(cwd, true, inferred_pkgid), &Everything);
                        }
                        None  => { usage::install(); return; }
                        Some((ws, pkgid))                => {
                            let pkg_src = PkgSrc::new(ws, false, pkgid);
                            self.install(pkg_src, &Everything);
                      }
                  }
                }
                else {
                    // The package id is presumed to be the first command-line
                    // argument
                    let pkgid = PkgId::new(args[0]);
                    let workspaces = pkg_parent_workspaces(&self.context, &pkgid);
                    debug2!("package ID = {}, found it in {:?} workspaces",
                           pkgid.to_str(), workspaces.len());
                    if workspaces.is_empty() {
                        let rp = rust_path();
                        assert!(!rp.is_empty());
                        let src = PkgSrc::new(rp[0].clone(), false, pkgid.clone());
                        self.install(src, &Everything);
                    }
                    else {
                        for workspace in workspaces.iter() {
                            let src = PkgSrc::new(workspace.clone(),
                                                  self.context.use_rust_path_hack,
                                                  pkgid.clone());
                            self.install(src, &Everything);
                        };
                    }
                }
            }
            "list" => {
                io::println("Installed packages:");
                do installed_packages::list_installed_packages |pkg_id| {
                    println(pkg_id.path.to_str());
                    true
                };
            }
            "prefer" => {
                if args.len() < 1 {
                    return usage::uninstall();
                }

                self.prefer(args[0], None);
            }
            "test" => {
                // Build the test executable
                let maybe_id_and_workspace = self.build_args(args, &Tests);
                match maybe_id_and_workspace {
                    Some((pkg_id, workspace)) => {
                        // Assuming it's built, run the tests
                        self.test(&pkg_id, &workspace);
                    }
                    None => {
                        error("Testing failed because building the specified package failed.");
                    }
                }
            }
            "init" => {
                if args.len() != 0 {
                    return usage::init();
                } else {
                    self.init();
                }
            }
            "uninstall" => {
                if args.len() < 1 {
                    return usage::uninstall();
                }

                let pkgid = PkgId::new(args[0]);
                if !installed_packages::package_is_installed(&pkgid) {
                    warn(format!("Package {} doesn't seem to be installed! \
                                  Doing nothing.", args[0]));
                    return;
                }
                else {
                    let rp = rust_path();
                    assert!(!rp.is_empty());
                    do each_pkg_parent_workspace(&self.context, &pkgid) |workspace| {
                        path_util::uninstall_package_from(workspace, &pkgid);
                        note(format!("Uninstalled package {} (was installed in {})",
                                  pkgid.to_str(), workspace.to_str()));
                        true
                    };
                }
            }
            "unprefer" => {
                if args.len() < 1 {
                    return usage::unprefer();
                }

                self.unprefer(args[0], None);
            }
            _ => fail2!("I don't know the command `{}`", cmd)
        }
    }

    fn do_cmd(&self, _cmd: &str, _pkgname: &str)  {
        // stub
        fail2!("`do` not yet implemented");
    }

    /// Returns the destination workspace
    /// In the case of a custom build, we don't know, so we just return the source workspace
    /// what_to_build says: "Just build the lib.rs file in one subdirectory,
    /// don't walk anything recursively." Or else, everything.
    fn build(&self, pkg_src: &mut PkgSrc, what_to_build: &WhatToBuild) -> Path {
        let workspace = pkg_src.workspace.clone();
        let pkgid = pkg_src.id.clone();

        debug2!("build: workspace = {} (in Rust path? {:?} is git dir? {:?} \
                pkgid = {} pkgsrc start_dir = {}", workspace.to_str(),
               in_rust_path(&workspace), is_git_dir(&workspace.push_rel(&pkgid.path)),
               pkgid.to_str(), pkg_src.start_dir.to_str());

        // If workspace isn't in the RUST_PATH, and it's a git repo,
        // then clone it into the first entry in RUST_PATH, and repeat
        if !in_rust_path(&workspace) && is_git_dir(&workspace.push_rel(&pkgid.path)) {
            let out_dir = default_workspace().push("src").push_rel(&pkgid.path);
            source_control::git_clone(&workspace.push_rel(&pkgid.path),
                                      &out_dir, &pkgid.version);
            let default_ws = default_workspace();
            debug2!("Calling build recursively with {:?} and {:?}", default_ws.to_str(),
                   pkgid.to_str());
            return self.build(&mut PkgSrc::new(default_ws, false, pkgid.clone()), what_to_build);
        }

        // Is there custom build logic? If so, use it
        let mut custom = false;
        debug2!("Package source directory = {}", pkg_src.to_str());
        let opt = pkg_src.package_script_option();
        debug2!("Calling pkg_script_option on {:?}", opt);
        let cfgs = match pkg_src.package_script_option() {
            Some(package_script_path) => {
                let sysroot = self.sysroot_to_use();
                let (cfgs, hook_result) =
                    do self.workcache_context.with_prep(package_script_path.to_str()) |prep| {
                    let sub_sysroot = sysroot.clone();
                    let package_script_path_clone = package_script_path.clone();
                    let sub_ws = workspace.clone();
                    let sub_id = pkgid.clone();
                    declare_package_script_dependency(prep, &*pkg_src);
                    do prep.exec |exec| {
                        let mut pscript = PkgScript::parse(@sub_sysroot.clone(),
                                                          package_script_path_clone.clone(),
                                                          &sub_ws,
                                                          &sub_id);

                        pscript.run_custom(exec, &sub_sysroot)
                    }
                };
                debug2!("Command return code = {:?}", hook_result);
                if hook_result != 0 {
                    fail2!("Error running custom build command")
                }
                custom = true;
                // otherwise, the package script succeeded
                cfgs
            }
            None => {
                debug2!("No package script, continuing");
                ~[]
            }
        } + self.context.cfgs;

        // If there was a package script, it should have finished
        // the build already. Otherwise...
        if !custom {
            match what_to_build {
                // Find crates inside the workspace
                &Everything => pkg_src.find_crates(),
                // Find only tests
                &Tests => pkg_src.find_crates_with_filter(|s| { is_test(&Path(s)) }),
                // Don't infer any crates -- just build the one that was requested
                &JustOne(ref p) => {
                    // We expect that p is relative to the package source's start directory,
                    // so check that assumption
                    debug2!("JustOne: p = {}", p.to_str());
                    assert!(os::path_exists(&pkg_src.start_dir.push_rel(p)));
                    if is_lib(p) {
                        PkgSrc::push_crate(&mut pkg_src.libs, 0, p);
                    } else if is_main(p) {
                        PkgSrc::push_crate(&mut pkg_src.mains, 0, p);
                    } else if is_test(p) {
                        PkgSrc::push_crate(&mut pkg_src.tests, 0, p);
                    } else if is_bench(p) {
                        PkgSrc::push_crate(&mut pkg_src.benchs, 0, p);
                    } else {
                        warn(format!("Not building any crates for dependency {}", p.to_str()));
                        return workspace.clone();
                    }
                }
            }
            // Build it!
            let rs_path = pkg_src.build(self, cfgs);
            Path(rs_path)
        }
        else {
            // Just return the source workspace
            workspace.clone()
        }
    }

    fn clean(&self, workspace: &Path, id: &PkgId)  {
        // Could also support a custom build hook in the pkg
        // script for cleaning files rustpkg doesn't know about.
        // Do something reasonable for now

        let dir = build_pkg_id_in_workspace(id, workspace);
        note(format!("Cleaning package {} (removing directory {})",
                        id.to_str(), dir.to_str()));
        if os::path_exists(&dir) {
            os::remove_dir_recursive(&dir);
            note(format!("Removed directory {}", dir.to_str()));
        }

        note(format!("Cleaned package {}", id.to_str()));
    }

    fn info(&self) {
        // stub
        fail2!("info not yet implemented");
    }

    fn install(&self, mut pkg_src: PkgSrc, what: &WhatToBuild) -> (~[Path], ~[(~str, ~str)]) {

        let id = pkg_src.id.clone();

        let mut installed_files = ~[];
        let inputs = ~[];

        // workcache only knows about *crates*. Building a package
        // just means inferring all the crates in it, then building each one.
        let destination_workspace = self.build(&mut pkg_src, what).to_str();

        let to_do = ~[pkg_src.libs.clone(), pkg_src.mains.clone(),
                      pkg_src.tests.clone(), pkg_src.benchs.clone()];
        debug2!("In declare inputs for {}", id.to_str());
        for cs in to_do.iter() {
            for c in cs.iter() {
                let path = pkg_src.start_dir.push_rel(&c.file).normalize();
                debug2!("Recording input: {}", path.to_str());
                installed_files.push(path);
            }
        }
        // See #7402: This still isn't quite right yet; we want to
        // install to the first workspace in the RUST_PATH if there's
        // a non-default RUST_PATH. This code installs to the same
        // workspace the package was built in.
        let actual_workspace = if path_util::user_set_rust_path() {
            default_workspace()
        }
            else {
            Path(destination_workspace)
        };
        debug2!("install: destination workspace = {}, id = {}, installing to {}",
               destination_workspace, id.to_str(), actual_workspace.to_str());
        let result = self.install_no_build(&Path(destination_workspace),
                                           &actual_workspace,
                                           &id).map(|s| Path(*s));
        debug2!("install: id = {}, about to call discover_outputs, {:?}",
               id.to_str(), result.to_str());
        installed_files = installed_files + result;
        note(format!("Installed package {} to {}", id.to_str(), actual_workspace.to_str()));
        (installed_files, inputs)
    }

    // again, working around lack of Encodable for Path
    fn install_no_build(&self,
                        source_workspace: &Path,
                        target_workspace: &Path,
                        id: &PkgId) -> ~[~str] {
        use conditions::copy_failed::cond;

        // Now copy stuff into the install dirs
        let maybe_executable = built_executable_in_workspace(id, source_workspace);
        let maybe_library = built_library_in_workspace(id, source_workspace);
        let target_exec = target_executable_in_workspace(id, target_workspace);
        let target_lib = maybe_library.map(|_p| target_library_in_workspace(id, target_workspace));

        debug2!("target_exec = {} target_lib = {:?} \
               maybe_executable = {:?} maybe_library = {:?}",
               target_exec.to_str(), target_lib,
               maybe_executable, maybe_library);

        do self.workcache_context.with_prep(id.install_tag()) |prep| {
            for ee in maybe_executable.iter() {
                prep.declare_input("binary",
                                   ee.to_str(),
                                   workcache_support::digest_only_date(ee));
            }
            for ll in maybe_library.iter() {
                prep.declare_input("binary",
                                   ll.to_str(),
                                   workcache_support::digest_only_date(ll));
            }
            let subex = maybe_executable.clone();
            let sublib = maybe_library.clone();
            let sub_target_ex = target_exec.clone();
            let sub_target_lib = target_lib.clone();

            do prep.exec |exe_thing| {
                let mut outputs = ~[];

                for exec in subex.iter() {
                    debug2!("Copying: {} -> {}", exec.to_str(), sub_target_ex.to_str());
                    if !(os::mkdir_recursive(&sub_target_ex.dir_path(), U_RWX) &&
                         os::copy_file(exec, &sub_target_ex)) {
                        cond.raise(((*exec).clone(), sub_target_ex.clone()));
                    }
                    exe_thing.discover_output("binary",
                        sub_target_ex.to_str(),
                        workcache_support::digest_only_date(&sub_target_ex));
                    outputs.push(sub_target_ex.to_str());
                }
                for lib in sublib.iter() {
                    let target_lib = sub_target_lib
                        .clone().expect(format!("I built {} but apparently \
                                             didn't install it!", lib.to_str()));
                    let target_lib = target_lib
                        .pop().push(lib.filename().expect("weird target lib"));
                    debug2!("Copying: {} -> {}", lib.to_str(), sub_target_lib.to_str());
                    if !(os::mkdir_recursive(&target_lib.dir_path(), U_RWX) &&
                         os::copy_file(lib, &target_lib)) {
                        cond.raise(((*lib).clone(), target_lib.clone()));
                    }
                    exe_thing.discover_output("binary",
                                              target_lib.to_str(),
                                              workcache_support::digest_only_date(&target_lib));
                    outputs.push(target_lib.to_str());
                }
                outputs
            }
        }
    }

    fn prefer(&self, _id: &str, _vers: Option<~str>)  {
        fail2!("prefer not yet implemented");
    }

    fn test(&self, pkgid: &PkgId, workspace: &Path)  {
        match built_test_in_workspace(pkgid, workspace) {
            Some(test_exec) => {
                debug2!("test: test_exec = {}", test_exec.to_str());
                let status = run::process_status(test_exec.to_str(), [~"--test"]);
                os::set_exit_status(status);
            }
            None => {
                error(format!("Internal error: test executable for package ID {} in workspace {} \
                           wasn't built! Please report this as a bug.",
                           pkgid.to_str(), workspace.to_str()));
            }
        }
    }

    fn init(&self) {
        os::mkdir_recursive(&Path("src"),   U_RWX);
        os::mkdir_recursive(&Path("lib"),   U_RWX);
        os::mkdir_recursive(&Path("bin"),   U_RWX);
        os::mkdir_recursive(&Path("build"), U_RWX);
    }

    fn uninstall(&self, _id: &str, _vers: Option<~str>)  {
        fail2!("uninstall not yet implemented");
    }

    fn unprefer(&self, _id: &str, _vers: Option<~str>)  {
        fail2!("unprefer not yet implemented");
    }
}

pub fn main() {
    io::println("WARNING: The Rust package manager is experimental and may be unstable");
    os::set_exit_status(main_args(os::args()));
}

pub fn main_args(args: &[~str]) -> int {
    let opts = ~[getopts::optflag("h"), getopts::optflag("help"),
                                        getopts::optflag("no-link"),
                                        getopts::optflag("no-trans"),
                 // n.b. Ignores different --pretty options for now
                                        getopts::optflag("pretty"),
                                        getopts::optflag("parse-only"),
                 getopts::optflag("S"), getopts::optflag("assembly"),
                 getopts::optmulti("c"), getopts::optmulti("cfg"),
                 getopts::optflag("v"), getopts::optflag("version"),
                 getopts::optflag("r"), getopts::optflag("rust-path-hack"),
                                        getopts::optopt("sysroot"),
                                        getopts::optflag("emit-llvm"),
                                        getopts::optopt("linker"),
                                        getopts::optopt("link-args"),
                                        getopts::optopt("opt-level"),
                 getopts::optflag("O"),
                                        getopts::optflag("save-temps"),
                                        getopts::optopt("target"),
                                        getopts::optopt("target-cpu"),
                 getopts::optmulti("Z")                                   ];
    let matches = &match getopts::getopts(args, opts) {
        result::Ok(m) => m,
        result::Err(f) => {
            error(format!("{}", f.to_err_msg()));

            return 1;
        }
    };
    let mut help = matches.opt_present("h") ||
                   matches.opt_present("help");
    let no_link = matches.opt_present("no-link");
    let no_trans = matches.opt_present("no-trans");
    let supplied_sysroot = matches.opt_val("sysroot");
    let generate_asm = matches.opt_present("S") ||
        matches.opt_present("assembly");
    let parse_only = matches.opt_present("parse-only");
    let pretty = matches.opt_present("pretty");
    let emit_llvm = matches.opt_present("emit-llvm");

    if matches.opt_present("v") ||
       matches.opt_present("version") {
        rustc::version(args[0]);
        return 0;
    }

    let use_rust_path_hack = matches.opt_present("r") ||
                             matches.opt_present("rust-path-hack");

    let linker = matches.opt_str("linker");
    let link_args = matches.opt_str("link-args");
    let cfgs = matches.opt_strs("cfg") + matches.opt_strs("c");
    let mut user_supplied_opt_level = true;
    let opt_level = match matches.opt_str("opt-level") {
        Some(~"0") => session::No,
        Some(~"1") => session::Less,
        Some(~"2") => session::Default,
        Some(~"3") => session::Aggressive,
        _ if matches.opt_present("O") => session::Default,
        _ => {
            user_supplied_opt_level = false;
            session::No
        }
    };

    let save_temps = matches.opt_present("save-temps");
    let target     = matches.opt_str("target");
    let target_cpu = matches.opt_str("target-cpu");
    let experimental_features = {
        let strs = matches.opt_strs("Z");
        if matches.opt_present("Z") {
            Some(strs)
        }
        else {
            None
        }
    };

    let mut args = matches.free.clone();
    args.shift();

    if (args.len() < 1) {
        usage::general();
        return 1;
    }

    let rustc_flags = RustcFlags {
        linker: linker,
        link_args: link_args,
        optimization_level: opt_level,
        compile_upto: if no_trans {
            Trans
        } else if no_link {
            Link
        } else if pretty {
            Pretty
        } else if parse_only {
            Analysis
        } else if emit_llvm && generate_asm {
            LLVMAssemble
        } else if generate_asm {
            Assemble
        } else if emit_llvm {
            LLVMCompileBitcode
        } else {
            Nothing
        },
        save_temps: save_temps,
        target: target,
        target_cpu: target_cpu,
        experimental_features: experimental_features
    };

    let mut cmd_opt = None;
    for a in args.iter() {
        if util::is_cmd(*a) {
            cmd_opt = Some(a);
            break;
        }
    }
    let cmd = match cmd_opt {
        None => {
            usage::general();
            return 0;
        }
        Some(cmd) => {
            help |= context::flags_ok_for_cmd(&rustc_flags, cfgs, *cmd, user_supplied_opt_level);
            if help {
                match *cmd {
                    ~"build" => usage::build(),
                    ~"clean" => usage::clean(),
                    ~"do" => usage::do_cmd(),
                    ~"info" => usage::info(),
                    ~"install" => usage::install(),
                    ~"list"    => usage::list(),
                    ~"prefer" => usage::prefer(),
                    ~"test" => usage::test(),
                    ~"init" => usage::init(),
                    ~"uninstall" => usage::uninstall(),
                    ~"unprefer" => usage::unprefer(),
                    _ => usage::general()
                };
                return 0;
            } else {
                cmd
            }
        }
    };

    // Pop off all flags, plus the command
    let remaining_args = args.iter().skip_while(|s| !util::is_cmd(**s));
    // I had to add this type annotation to get the code to typecheck
    let mut remaining_args: ~[~str] = remaining_args.map(|s| (*s).clone()).collect();
    remaining_args.shift();
    let sroot = match supplied_sysroot {
        Some(getopts::Val(s)) => Path(s),
        _ => filesearch::get_or_default_sysroot()
    };

    debug2!("Using sysroot: {}", sroot.to_str());
    debug2!("Will store workcache in {}", default_workspace().to_str());

    let rm_args = remaining_args.clone();
    let sub_cmd = cmd.clone();
    // Wrap the rest in task::try in case of a condition failure in a task
    let result = do task::try {
        BuildContext {
            context: Context {
                cfgs: cfgs.clone(),
                rustc_flags: rustc_flags.clone(),
                use_rust_path_hack: use_rust_path_hack,
                sysroot: sroot.clone(), // Currently, only tests override this
            },
            workcache_context: api::default_context(default_workspace()).workcache_context
        }.run(sub_cmd, rm_args.clone())
    };
    // FIXME #9262: This is using the same error code for all errors,
    // and at least one test case succeeds if rustpkg returns COPY_FAILED_CODE,
    // when actually, it might set the exit code for that even if a different
    // unhandled condition got raised.
    if result.is_err() { return COPY_FAILED_CODE; }
    return 0;
}

/**
 * Get the working directory of the package script.
 * Assumes that the package script has been compiled
 * in is the working directory.
 */
pub fn work_dir() -> Path {
    os::self_exe_path().unwrap()
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

fn declare_package_script_dependency(prep: &mut workcache::Prep, pkg_src: &PkgSrc) {
    match pkg_src.package_script_option() {
        Some(ref p) => prep.declare_input("file", p.to_str(),
                                      workcache_support::digest_file_with_date(p)),
        None => ()
    }
}
