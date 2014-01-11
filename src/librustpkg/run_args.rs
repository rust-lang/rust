// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{os, run};
use std::io;
use std::io::fs;
pub use std::path::Path;
use rustc::metadata::filesearch::rust_path;

use crate_id::CrateId;
use context::{Command, BuildContext};
use installed_packages;
use messages::{error, warn, note};
use package_source::PkgSrc;
use path_util::{build_pkg_id_in_workspace, built_test_in_workspace};
use path_util::{default_workspace, built_library_in_workspace};
use path_util::{built_executable_in_workspace};
use path_util::{target_executable_in_workspace};
use path_util::{target_library_in_workspace, dir_has_crate_file};
use path_util::{in_rust_path, uninstall_package_from};
use PkgScript;
use source_control;
use source_control::{CheckedOutSources, is_git_dir, make_read_only};
use target::{WhatToBuild, Everything, is_lib, is_main, is_test, is_bench};
use target::{Tests, MaybeCustom, Inferred, JustOne};
use usage;
use workspace::{each_pkg_parent_workspace, pkg_parent_workspaces, cwd_to_workspace};
use workspace::determine_destination;
use workcache_support;

/// Perform an action based on command line input
pub fn run_args(bctx: &BuildContext, cmd: Command, args: ~[~str]) {
    let cwd = os::getcwd();
    match cmd {
        BuildCmd => {
            build_args(bctx, args, &WhatToBuild::new(MaybeCustom, Everything));
        }
        CleanCmd => {
            if args.len() < 1 {
                match cwd_to_workspace() {
                    None => { usage::clean(); return }
                    // tjc: Maybe clean should clean all the packages in the
                    // current workspace, though?
                    Some((ws, crateid)) => clean(&ws, &crateid)
                }

            }
            else {
                // The package id is presumed to be the first command-line
                // argument
                let crateid = CrateId::new(args[0].clone());
                clean(&cwd, &crateid); // tjc: should use workspace, not cwd
            }
        }
        DoCmd => {
            if args.len() < 2 {
                return usage::do_cmd();
            }

            do_cmd(args[0].clone(), args[1].clone());
        }
        InfoCmd => {
            info();
        }
        InstallCmd => {
           if args.len() < 1 {
                match cwd_to_workspace() {
                    None if dir_has_crate_file(&cwd) => {
                        // FIXME (#9639): This needs to handle non-utf8 paths

                        let inferred_crateid =
                            CrateId::new(cwd.filename_str().unwrap());
                        install(bctx,
                                PkgSrc::new(cwd, default_workspace(),
                                                 true, inferred_crateid),
                                &WhatToBuild::new(MaybeCustom, Everything));
                    }
                    None  => { usage::install(); return; }
                    Some((ws, crateid))                => {
                        let pkg_src = PkgSrc::new(ws.clone(), ws.clone(), false, crateid);
                        install(bctx,
                                pkg_src,
                                &WhatToBuild::new(MaybeCustom, Everything));
                  }
              }
            }
            else {
                // The package id is presumed to be the first command-line
                // argument
                let crateid = CrateId::new(args[0]);
                let workspaces = pkg_parent_workspaces(
                        &bctx.context, &crateid);
                debug!("package ID = {}, found it in {:?} workspaces",
                       crateid.to_str(), workspaces.len());
                if workspaces.is_empty() {
                    let d = default_workspace();
                    let src = PkgSrc::new(d.clone(), d, false, crateid.clone());
                    install(bctx,
                            src,
                            &WhatToBuild::new(MaybeCustom, Everything));
                }
                else {
                    for workspace in workspaces.iter() {
                        let dest = determine_destination(os::getcwd(),
                                         bctx.context.use_rust_path_hack,
                                         workspace);
                        let src = PkgSrc::new(workspace.clone(),
                                         dest,
                                         bctx.context.use_rust_path_hack,
                                         crateid.clone());
                        install(bctx,
                                src,
                                &WhatToBuild::new(MaybeCustom, Everything));
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
                return usage::uninstall();
            }

            prefer(args[0], None);
        }
        TestCmd => {
            // Build the test executable
            let maybe_id_and_workspace = build_args(bctx,args,
                                        &WhatToBuild::new(MaybeCustom, Tests));
            match maybe_id_and_workspace {
                Some((pkg_id, workspace)) => {
                    // Assuming it's built, run the tests
                    test(&pkg_id, &workspace);
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
                init();
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
                each_pkg_parent_workspace(&bctx.context, &crateid, |workspace| {
                    uninstall_package_from(workspace, &crateid);
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

            unprefer(args[0], None);
        }
    }
}

/// Returns a pair of the selected package ID, and the destination workspace
fn build_args(bctx: &BuildContext,
              args: ~[~str],
              what: &WhatToBuild) -> Option<(CrateId, Path)> {
    let cwd = os::getcwd();

    if args.len() < 1 {
        match cwd_to_workspace() {
            None  if dir_has_crate_file(&cwd) => {
                // FIXME (#9639): This needs to handle non-utf8 paths
                let crateid = CrateId::new(cwd.filename_str().unwrap());
                let mut pkg_src = PkgSrc::new(cwd, default_workspace(), true, crateid);
                build(bctx, &mut pkg_src, what);
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
                build(bctx, &mut pkg_src, what);
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
        each_pkg_parent_workspace(&bctx.context, &crateid, |workspace| {
            debug!("found pkg {} in workspace {}, trying to build",
                   crateid.to_str(), workspace.display());
            dest_ws = determine_destination(os::getcwd(),
                                    bctx.context.use_rust_path_hack,
                                    workspace);
            let mut pkg_src = PkgSrc::new(workspace.clone(), dest_ws.clone(),
                                          false, crateid.clone());
            build(bctx, &mut pkg_src, what);
            true
        });
        // n.b. If this builds multiple packages, it only returns the workspace for
        // the last one. The whole building-multiple-packages-with-the-same-ID is weird
        // anyway and there are no tests for it, so maybe take it out
        Some((crateid, dest_ws))
    }
}

/// Perform "do cmd" based on parsed command line input
fn do_cmd(_cmd: &str, _pkgname: &str)  {
    // stub
    fail!("`do` not yet implemented");
}

/// Returns the destination workspace
fn build(bctx: &BuildContext, pkg_src: &mut PkgSrc, what_to_build: &WhatToBuild) {
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
        return build(bctx,
                         &mut PkgSrc::new(default_ws.clone(),
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
            let sysroot = bctx.sysroot_to_use();
            // Build the package script if needed
            let script_build = format!("build_package_script({})",
                                       package_script_path.display());
            let pkg_exe = bctx.workcache_context.with_prep(script_build, |prep| {
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
    } + bctx.context.cfgs;

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
        pkg_src.build(bctx, cfgs, []);
    }
}

/// Perform "clean" based on command line input
fn clean(workspace: &Path, id: &CrateId)  {
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

/// Perform "info" based on command line input
fn info() {
    // stub
    fail!("info not yet implemented");
}

/// Returns a pair. First component is a list of installed paths,
/// second is a list of declared and discovered inputs
pub fn install(bctx: &BuildContext,
               mut pkg_src: PkgSrc,
               what: &WhatToBuild) -> (~[Path], ~[(~str, ~str)]) {

    let id = pkg_src.id.clone();

    let mut installed_files = ~[];
    let mut inputs = ~[];
    let mut build_inputs = ~[];

    debug!("Installing package source: {}", pkg_src.to_str());

    // workcache only knows about *crates*. Building a package
    // just means inferring all the crates in it, then building each one.
    build(bctx, &mut pkg_src, what);

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

    let result = install_no_build(bctx,
                                  pkg_src.build_workspace(),
                                  build_inputs,
                                  &pkg_src.destination_workspace,
                                  &id).map(|s| Path::new(s.as_slice()));
    installed_files = installed_files + result;
    note(format!("Installed package {} to {}",
                 id.to_str(),
                 pkg_src.destination_workspace.display()));
    (installed_files, inputs)
}

/// Working around lack of Encodable for Path
fn install_no_build(bctx: &BuildContext,
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

    bctx.workcache_context.with_prep(id.install_tag(), |prep| {
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

///  Run "prefer" based on command line input
fn prefer(_id: &str, _vers: Option<~str>)  {
    fail!("prefer not yet implemented");
}

///  Run "build" based on command line input
fn test(crateid: &CrateId, workspace: &Path)  {
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

fn init() {
    fs::mkdir_recursive(&Path::new("src"), io::UserRWX);
    fs::mkdir_recursive(&Path::new("bin"), io::UserRWX);
    fs::mkdir_recursive(&Path::new("lib"), io::UserRWX);
    fs::mkdir_recursive(&Path::new("build"), io::UserRWX);
}

fn unprefer(_id: &str, _vers: Option<~str>)  {
    fail!("unprefer not yet implemented");
}
