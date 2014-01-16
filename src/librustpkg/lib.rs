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

#[crate_id = "rustpkg#0.10-pre"];
#[license = "MIT/ASL2"];
#[crate_type = "dylib"];

#[feature(globs, managed_boxes)];

extern mod extra;
extern mod rustc;
extern mod syntax;

use std::{os, run, str, task};
use std::io::process;
use std::io;
use std::io::fs;
pub use std::path::Path;

use extra::workcache;
use rustc::driver::{driver, session};
use rustc::metadata::creader::Loader;
use rustc::metadata::filesearch;
use rustc::metadata::filesearch::rust_path;
use rustc::util::sha2;
use syntax::{ast, diagnostic};
use messages::{error, warn, note};
use parse_args::{ParseResult, parse_args};
use path_util::{build_pkg_id_in_workspace, built_test_in_workspace};
use path_util::in_rust_path;
use path_util::{built_executable_in_workspace, built_library_in_workspace, default_workspace};
use path_util::{target_executable_in_workspace, target_library_in_workspace, dir_has_crate_file};
use source_control::{CheckedOutSources, is_git_dir, make_read_only};
use workspace::{each_pkg_parent_workspace, pkg_parent_workspaces, cwd_to_workspace};
use workspace::determine_destination;
use context::{BuildContext, Trans, Nothing, Pretty, Analysis,
              LLVMAssemble, LLVMCompileBitcode};
use context::{Command, BuildCmd, CleanCmd, DoCmd, HelpCmd, InfoCmd, InstallCmd, ListCmd,
    PreferCmd, TestCmd, InitCmd, UninstallCmd, UnpreferCmd};
use crate_id::CrateId;
use package_source::PkgSrc;
use target::{WhatToBuild, Everything, is_lib, is_main, is_test, is_bench};
use target::{Main, Tests, MaybeCustom, Inferred, JustOne};
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
        let loader = &mut Loader::new(sess);
        let crate_and_map = driver::phase_2_configure_and_expand(sess,
                                                         cfg.clone(),
                                                         loader,
                                                         crate);
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

pub trait CtxMethods {
    fn run(&self, cmd: Command, args: ~[~str]);
    fn do_cmd(&self, _cmd: &str, _pkgname: &str);
    /// Returns a pair of the selected package ID, and the destination workspace
    fn build_args(&self, args: ~[~str], what: &WhatToBuild) -> Option<(CrateId, Path)>;
    /// Returns the destination workspace
    fn build(&self, pkg_src: &mut PkgSrc, what: &WhatToBuild);
    fn clean(&self, workspace: &Path, id: &CrateId);
    fn info(&self);
    /// Returns a pair. First component is a list of installed paths,
    /// second is a list of declared and discovered inputs
    fn install(&self, src: PkgSrc, what: &WhatToBuild) -> (~[Path], ~[(~str, ~str)]);
    /// Returns a list of installed files
    fn install_no_build(&self,
                        build_workspace: &Path,
                        build_inputs: &[Path],
                        target_workspace: &Path,
                        id: &CrateId) -> ~[~str];
    fn prefer(&self, _id: &str, _vers: Option<~str>);
    fn test(&self, id: &CrateId, workspace: &Path);
    fn uninstall(&self, _id: &str, _vers: Option<~str>);
    fn unprefer(&self, _id: &str, _vers: Option<~str>);
    fn init(&self);
}

impl CtxMethods for BuildContext {
    fn build_args(&self, args: ~[~str], what: &WhatToBuild) -> Option<(CrateId, Path)> {
        let cwd = os::getcwd();

        if args.len() < 1 {
            match cwd_to_workspace() {
                None  if dir_has_crate_file(&cwd) => {
                    // FIXME (#9639): This needs to handle non-utf8 paths
                    let crateid = CrateId::new(cwd.filename_str().unwrap());
                    let mut pkg_src = PkgSrc::new(cwd, default_workspace(), true, crateid);
                    self.build(&mut pkg_src, what);
                    match pkg_src {
                        PkgSrc { destination_workspace: ws,
                                 id: id, .. } => {
                            Some((id, ws))
                        }
                    }
                }
                None => { usage::build(); None }
                Some((ws, crateid)) => {
                    let mut pkg_src = PkgSrc::new(ws.clone(), ws, false, crateid);
                    self.build(&mut pkg_src, what);
                    match pkg_src {
                        PkgSrc { destination_workspace: ws,
                                 id: id, .. } => {
                            Some((id, ws))
                        }
                    }
                }
            }
        } else {
            // The package id is presumed to be the first command-line
            // argument
            let crateid = CrateId::new(args[0].clone());
            let mut dest_ws = default_workspace();
            each_pkg_parent_workspace(&self.context, &crateid, |workspace| {
                debug!("found pkg {} in workspace {}, trying to build",
                       crateid.to_str(), workspace.display());
                dest_ws = determine_destination(os::getcwd(),
                                                self.context.use_rust_path_hack,
                                                workspace);
                let mut pkg_src = PkgSrc::new(workspace.clone(), dest_ws.clone(),
                                              false, crateid.clone());
                self.build(&mut pkg_src, what);
                true
            });
            // n.b. If this builds multiple packages, it only returns the workspace for
            // the last one. The whole building-multiple-packages-with-the-same-ID is weird
            // anyway and there are no tests for it, so maybe take it out
            Some((crateid, dest_ws))
        }
    }
    fn run(&self, cmd: Command, args: ~[~str]) {
        let cwd = os::getcwd();
        match cmd {
            BuildCmd => {
                self.build_args(args, &WhatToBuild::new(MaybeCustom, Everything));
            }
            CleanCmd => {
                if args.len() < 1 {
                    match cwd_to_workspace() {
                        None => { usage::clean(); return }
                        // tjc: Maybe clean should clean all the packages in the
                        // current workspace, though?
                        Some((ws, crateid)) => self.clean(&ws, &crateid)
                    }

                }
                else {
                    // The package id is presumed to be the first command-line
                    // argument
                    let crateid = CrateId::new(args[0].clone());
                    self.clean(&cwd, &crateid); // tjc: should use workspace, not cwd
                }
            }
            DoCmd => {
                if args.len() < 2 {
                    return usage::do_cmd();
                }

                self.do_cmd(args[0].clone(), args[1].clone());
            }
            HelpCmd => {
                if args.len() != 1 {
                    return usage::general();
                }
                match FromStr::from_str(args[0]) {
                    Some(help_cmd) => usage::usage_for_command(help_cmd),
                    None => {
                        usage::general();
                        error(format!("{} is not a recognized command", args[0]))
                    }
                }
            }
            InfoCmd => {
                self.info();
            }
            InstallCmd => {
               if args.len() < 1 {
                    match cwd_to_workspace() {
                        None if dir_has_crate_file(&cwd) => {
                            // FIXME (#9639): This needs to handle non-utf8 paths

                            let inferred_crateid =
                                CrateId::new(cwd.filename_str().unwrap());
                            self.install(PkgSrc::new(cwd, default_workspace(),
                                                     true, inferred_crateid),
                                         &WhatToBuild::new(MaybeCustom, Everything));
                        }
                        None  => { usage::install(); return; }
                        Some((ws, crateid))                => {
                            let pkg_src = PkgSrc::new(ws.clone(), ws.clone(), false, crateid);
                            self.install(pkg_src, &WhatToBuild::new(MaybeCustom,
                                                                    Everything));
                      }
                  }
                }
                else {
                    // The package id is presumed to be the first command-line
                    // argument
                    let crateid = CrateId::new(args[0]);
                    let workspaces = pkg_parent_workspaces(&self.context, &crateid);
                    debug!("package ID = {}, found it in {:?} workspaces",
                           crateid.to_str(), workspaces.len());
                    if workspaces.is_empty() {
                        let d = default_workspace();
                        let src = PkgSrc::new(d.clone(), d, false, crateid.clone());
                        self.install(src, &WhatToBuild::new(MaybeCustom, Everything));
                    }
                    else {
                        for workspace in workspaces.iter() {
                            let dest = determine_destination(os::getcwd(),
                                                             self.context.use_rust_path_hack,
                                                             workspace);
                            let src = PkgSrc::new(workspace.clone(),
                                                  dest,
                                                  self.context.use_rust_path_hack,
                                                  crateid.clone());
                            self.install(src, &WhatToBuild::new(MaybeCustom, Everything));
                        };
                    }
                }
            }
            ListCmd => {
                println!("Installed packages:");
                installed_packages::list_installed_packages(|pkg_id| {
                    pkg_id.path.display().with_str(|s| println!("{}", s));
                    true
                });
            }
            PreferCmd => {
                if args.len() < 1 {
                    return usage::prefer();
                }

                self.prefer(args[0], None);
            }
            TestCmd => {
                // Build the test executable
                let maybe_id_and_workspace = self.build_args(args,
                                                             &WhatToBuild::new(MaybeCustom, Tests));
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
            InitCmd => {
                if args.len() != 0 {
                    return usage::init();
                } else {
                    self.init();
                }
            }
            UninstallCmd => {
                if args.len() < 1 {
                    return usage::uninstall();
                }

                let crateid = CrateId::new(args[0]);
                if !installed_packages::package_is_installed(&crateid) {
                    warn(format!("Package {} doesn't seem to be installed! \
                                  Doing nothing.", args[0]));
                    return;
                }
                else {
                    let rp = rust_path();
                    assert!(!rp.is_empty());
                    each_pkg_parent_workspace(&self.context, &crateid, |workspace| {
                        path_util::uninstall_package_from(workspace, &crateid);
                        note(format!("Uninstalled package {} (was installed in {})",
                                  crateid.to_str(), workspace.display()));
                        true
                    });
                }
            }
            UnpreferCmd => {
                if args.len() < 1 {
                    return usage::unprefer();
                }

                self.unprefer(args[0], None);
            }
        }
    }

    fn do_cmd(&self, _cmd: &str, _pkgname: &str)  {
        // stub
        fail!("`do` not yet implemented");
    }

    fn build(&self, pkg_src: &mut PkgSrc, what_to_build: &WhatToBuild) {
        use conditions::git_checkout_failed::cond;

        let workspace = pkg_src.source_workspace.clone();
        let crateid = pkg_src.id.clone();

        debug!("build: workspace = {} (in Rust path? {:?} is git dir? {:?} \
                crateid = {} pkgsrc start_dir = {}", workspace.display(),
               in_rust_path(&workspace), is_git_dir(&workspace.join(&crateid.path)),
               crateid.to_str(), pkg_src.start_dir.display());
        debug!("build: what to build = {:?}", what_to_build);

        // If workspace isn't in the RUST_PATH, and it's a git repo,
        // then clone it into the first entry in RUST_PATH, and repeat
        if !in_rust_path(&workspace) && is_git_dir(&workspace.join(&crateid.path)) {
            let mut out_dir = default_workspace().join("src");
            out_dir.push(&crateid.path);
            let git_result = source_control::safe_git_clone(&workspace.join(&crateid.path),
                                                            &crateid.version,
                                                            &out_dir);
            match git_result {
                CheckedOutSources => make_read_only(&out_dir),
                // FIXME (#9639): This needs to handle non-utf8 paths
                _ => cond.raise((crateid.path.as_str().unwrap().to_owned(), out_dir.clone()))
            };
            let default_ws = default_workspace();
            debug!("Calling build recursively with {:?} and {:?}", default_ws.display(),
                   crateid.to_str());
            return self.build(&mut PkgSrc::new(default_ws.clone(),
                                               default_ws,
                                               false,
                                               crateid.clone()), what_to_build);
        }

        // Is there custom build logic? If so, use it
        let mut custom = false;
        debug!("Package source directory = {}", pkg_src.to_str());
        let opt = pkg_src.package_script_option();
        debug!("Calling pkg_script_option on {:?}", opt);
        let cfgs = match (pkg_src.package_script_option(), what_to_build.build_type) {
            (Some(package_script_path), MaybeCustom)  => {
                let sysroot = self.sysroot_to_use();
                // Build the package script if needed
                let script_build = format!("build_package_script({})",
                                           package_script_path.display());
                let pkg_exe = self.workcache_context.with_prep(script_build, |prep| {
                    let subsysroot = sysroot.clone();
                    let psp = package_script_path.clone();
                    let ws = workspace.clone();
                    let pid = crateid.clone();
                    prep.exec(proc(exec) {
                        let mut pscript = PkgScript::parse(subsysroot.clone(),
                                                           psp.clone(),
                                                           &ws,
                                                           &pid);
                        pscript.build_custom(exec)
                    })
                });
                // We always *run* the package script
                match PkgScript::run_custom(&Path::new(pkg_exe), &sysroot) {
                    Some((cfgs, hook_result)) => {
                        debug!("Command return code = {:?}", hook_result);
                        if !hook_result.success() {
                            fail!("Error running custom build command")
                        }
                        custom = true;
                        // otherwise, the package script succeeded
                        cfgs
                    },
                    None => {
                        fail!("Error starting custom build command")
                    }
                }
            }
            (Some(_), Inferred) => {
                debug!("There is a package script, but we're ignoring it");
                ~[]
            }
            (None, _) => {
                debug!("No package script, continuing");
                ~[]
            }
        } + self.context.cfgs;

        // If there was a package script, it should have finished
        // the build already. Otherwise...
        if !custom {
            match what_to_build.sources {
                // Find crates inside the workspace
                Everything => pkg_src.find_crates(),
                // Find only tests
                Tests => pkg_src.find_crates_with_filter(|s| { is_test(&Path::new(s)) }),
                // Don't infer any crates -- just build the one that was requested
                JustOne(ref p) => {
                    // We expect that p is relative to the package source's start directory,
                    // so check that assumption
                    debug!("JustOne: p = {}", p.display());
                    assert!(pkg_src.start_dir.join(p).exists());
                    if is_lib(p) {
                        PkgSrc::push_crate(&mut pkg_src.libs, 0, p);
                    } else if is_main(p) {
                        PkgSrc::push_crate(&mut pkg_src.mains, 0, p);
                    } else if is_test(p) {
                        PkgSrc::push_crate(&mut pkg_src.tests, 0, p);
                    } else if is_bench(p) {
                        PkgSrc::push_crate(&mut pkg_src.benchs, 0, p);
                    } else {
                        warn(format!("Not building any crates for dependency {}", p.display()));
                        return;
                    }
                }
            }
            // Build it!
            pkg_src.build(self, cfgs, []);
        }
    }

    fn clean(&self, workspace: &Path, id: &CrateId)  {
        // Could also support a custom build hook in the pkg
        // script for cleaning files rustpkg doesn't know about.
        // Do something reasonable for now

        let dir = build_pkg_id_in_workspace(id, workspace);
        note(format!("Cleaning package {} (removing directory {})",
                        id.to_str(), dir.display()));
        if dir.exists() {
            fs::rmdir_recursive(&dir);
            note(format!("Removed directory {}", dir.display()));
        }

        note(format!("Cleaned package {}", id.to_str()));
    }

    fn info(&self) {
        // stub
        fail!("info not yet implemented");
    }

    fn install(&self, mut pkg_src: PkgSrc, what: &WhatToBuild) -> (~[Path], ~[(~str, ~str)]) {

        let id = pkg_src.id.clone();

        let mut installed_files = ~[];
        let mut inputs = ~[];
        let mut build_inputs = ~[];

        debug!("Installing package source: {}", pkg_src.to_str());

        // workcache only knows about *crates*. Building a package
        // just means inferring all the crates in it, then building each one.
        self.build(&mut pkg_src, what);

        debug!("Done building package source {}", pkg_src.to_str());

        let to_do = ~[pkg_src.libs.clone(), pkg_src.mains.clone(),
                      pkg_src.tests.clone(), pkg_src.benchs.clone()];
        debug!("In declare inputs for {}", id.to_str());
        for cs in to_do.iter() {
            for c in cs.iter() {
                let path = pkg_src.start_dir.join(&c.file);
                debug!("Recording input: {}", path.display());
                // FIXME (#9639): This needs to handle non-utf8 paths
                inputs.push((~"file", path.as_str().unwrap().to_owned()));
                build_inputs.push(path);
            }
        }

        let result = self.install_no_build(pkg_src.build_workspace(),
                                           build_inputs,
                                           &pkg_src.destination_workspace,
                                           &id).map(|s| Path::new(s.as_slice()));
        installed_files = installed_files + result;
        note(format!("Installed package {} to {}",
                     id.to_str(),
                     pkg_src.destination_workspace.display()));
        (installed_files, inputs)
    }

    // again, working around lack of Encodable for Path
    fn install_no_build(&self,
                        build_workspace: &Path,
                        build_inputs: &[Path],
                        target_workspace: &Path,
                        id: &CrateId) -> ~[~str] {

        debug!("install_no_build: assuming {} comes from {} with target {}",
               id.to_str(), build_workspace.display(), target_workspace.display());

        // Now copy stuff into the install dirs
        let maybe_executable = built_executable_in_workspace(id, build_workspace);
        let maybe_library = built_library_in_workspace(id, build_workspace);
        let target_exec = target_executable_in_workspace(id, target_workspace);
        let target_lib = maybe_library.as_ref()
            .map(|_| target_library_in_workspace(id, target_workspace));

        debug!("target_exec = {} target_lib = {:?} \
               maybe_executable = {:?} maybe_library = {:?}",
               target_exec.display(), target_lib,
               maybe_executable, maybe_library);

        self.workcache_context.with_prep(id.install_tag(), |prep| {
            for ee in maybe_executable.iter() {
                // FIXME (#9639): This needs to handle non-utf8 paths
                prep.declare_input("binary",
                                   ee.as_str().unwrap(),
                                   workcache_support::digest_only_date(ee));
            }
            for ll in maybe_library.iter() {
                // FIXME (#9639): This needs to handle non-utf8 paths
                prep.declare_input("binary",
                                   ll.as_str().unwrap(),
                                   workcache_support::digest_only_date(ll));
            }
            let subex = maybe_executable.clone();
            let sublib = maybe_library.clone();
            let sub_target_ex = target_exec.clone();
            let sub_target_lib = target_lib.clone();
            let sub_build_inputs = build_inputs.to_owned();
            prep.exec(proc(exe_thing) {
                let mut outputs = ~[];
                // Declare all the *inputs* to the declared input too, as inputs
                for executable in subex.iter() {
                    exe_thing.discover_input("binary",
                                             executable.as_str().unwrap().to_owned(),
                                             workcache_support::digest_only_date(executable));
                }
                for library in sublib.iter() {
                    exe_thing.discover_input("binary",
                                             library.as_str().unwrap().to_owned(),
                                             workcache_support::digest_only_date(library));
                }

                for transitive_dependency in sub_build_inputs.iter() {
                    exe_thing.discover_input(
                        "file",
                        transitive_dependency.as_str().unwrap().to_owned(),
                        workcache_support::digest_file_with_date(transitive_dependency));
                }


                for exec in subex.iter() {
                    debug!("Copying: {} -> {}", exec.display(), sub_target_ex.display());
                    fs::mkdir_recursive(&sub_target_ex.dir_path(), io::UserRWX);
                    fs::copy(exec, &sub_target_ex);
                    // FIXME (#9639): This needs to handle non-utf8 paths
                    exe_thing.discover_output("binary",
                        sub_target_ex.as_str().unwrap(),
                        workcache_support::digest_only_date(&sub_target_ex));
                    outputs.push(sub_target_ex.as_str().unwrap().to_owned());
                }
                for lib in sublib.iter() {
                    let mut target_lib = sub_target_lib
                        .clone().expect(format!("I built {} but apparently \
                                             didn't install it!", lib.display()));
                    target_lib.set_filename(lib.filename().expect("weird target lib"));
                    fs::mkdir_recursive(&target_lib.dir_path(), io::UserRWX);
                    fs::copy(lib, &target_lib);
                    debug!("3. discovering output {}", target_lib.display());
                    exe_thing.discover_output("binary",
                                              target_lib.as_str().unwrap(),
                                              workcache_support::digest_only_date(&target_lib));
                    outputs.push(target_lib.as_str().unwrap().to_owned());
                }
                outputs
            })
        })
    }

    fn prefer(&self, _id: &str, _vers: Option<~str>)  {
        fail!("prefer not yet implemented");
    }

    fn test(&self, crateid: &CrateId, workspace: &Path)  {
        match built_test_in_workspace(crateid, workspace) {
            Some(test_exec) => {
                debug!("test: test_exec = {}", test_exec.display());
                // FIXME (#9639): This needs to handle non-utf8 paths
                let opt_status = run::process_status(test_exec.as_str().unwrap(), [~"--test"]);
                match opt_status {
                    Some(status) => {
                        if !status.success() {
                            fail!("Some tests failed");
                        }
                    },
                    None => fail!("Could not exec `{}`", test_exec.display())
                }
            }
            None => {
                error(format!("Internal error: test executable for package ID {} in workspace {} \
                           wasn't built! Please report this as a bug.",
                           crateid.to_str(), workspace.display()));
            }
        }
    }

    fn init(&self) {
        fs::mkdir_recursive(&Path::new("src"), io::UserRWX);
        fs::mkdir_recursive(&Path::new("bin"), io::UserRWX);
        fs::mkdir_recursive(&Path::new("lib"), io::UserRWX);
        fs::mkdir_recursive(&Path::new("build"), io::UserRWX);
    }

    fn uninstall(&self, _id: &str, _vers: Option<~str>)  {
        fail!("uninstall not yet implemented");
    }

    fn unprefer(&self, _id: &str, _vers: Option<~str>)  {
        fail!("unprefer not yet implemented");
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
        BuildContext {
            context: context,
            sysroot: sysroot.clone(), // Currently, only tests override this
            workcache_context: api::default_context(sysroot.clone(),
                                                    default_workspace()).workcache_context
        }.run(command, args.clone())
    };
    // FIXME #9262: This is using the same error code for all errors,
    // and at least one test case succeeds if rustpkg returns COPY_FAILED_CODE,
    // when actually, it might set the exit code for that even if a different
    // unhandled condition got raised.
    if result.is_err() { return COPY_FAILED_CODE; }
    return 0;
}
