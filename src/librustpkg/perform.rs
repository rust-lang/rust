// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This  module contains methods such as build and install. They do the actual
// work of rustpkg

use std::{io};
use std::io::fs;
pub use std::path::Path;

use context::{BuildContext};
use crate_id::{CrateId};
use messages::{warn, note};
use path_util::{in_rust_path, built_executable_in_workspace};
use path_util::{built_library_in_workspace, default_workspace};
use path_util::{target_executable_in_workspace, target_library_in_workspace};
use package_script::PkgScript;
use package_source::PkgSrc;
use source_control;
use source_control::{CheckedOutSources, is_git_dir, make_read_only};
use target::{WhatToBuild, Everything, is_lib, is_main, is_test, is_bench};
use target::{Tests, MaybeCustom, Inferred, JustOne};
use workcache_support::digest_only_date;
use workcache_support;


/// Build a package
pub fn build(pkg_src: &mut PkgSrc,
         what_to_build: &WhatToBuild,
         build_context: &BuildContext) {
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
        let pkg_src = &mut PkgSrc::new(default_ws.clone(),
                                       default_ws,
                                       false,
                                       crateid.clone());
        return build(pkg_src, what_to_build, build_context);
    }

    // Is there custom build logic? If so, use it
    let mut custom = false;
    debug!("Package source directory = {}", pkg_src.to_str());
    let opt = pkg_src.package_script_option();
    debug!("Calling pkg_script_option on {:?}", opt);
    let cfgs = match (pkg_src.package_script_option(), what_to_build.build_type) {
        (Some(package_script_path), MaybeCustom)  => {
            let sysroot = build_context.sysroot_to_use();
            // Build the package script if needed
            let script_build = format!("build_package_script({})",
                                       package_script_path.display());
            let pkg_exe = build_context.workcache_context.with_prep(script_build, |prep| {
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
    } + build_context.context.cfgs;

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
        pkg_src.build(build_context, cfgs, []);
    }
}

/// Install a package
pub fn install(mut pkg_src: PkgSrc,
           what: &WhatToBuild,
           build_context: &BuildContext) -> (~[Path], ~[(~str, ~str)]) {

    let id = pkg_src.id.clone();

    let mut installed_files = ~[];
    let mut inputs = ~[];
    let mut build_inputs = ~[];

    debug!("Installing package source: {}", pkg_src.to_str());

    // workcache only knows about *crates*. Building a package
    // just means inferring all the crates in it, then building each one.
    build(&mut pkg_src, what, build_context);

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

    let result =
          install_no_build(pkg_src.build_workspace(),
                           build_inputs,
                           &pkg_src.destination_workspace,
                           &id,
                           build_context).map(|s| Path::new(s.as_slice()));
    installed_files = installed_files + result;
    note(format!("Installed package {} to {}",
                 id.to_str(),
                 pkg_src.destination_workspace.display()));
    (installed_files, inputs)
}


/// Installing a package without building
fn install_no_build(build_workspace: &Path,
                    build_inputs: &[Path],
                    target_workspace: &Path,
                    id: &CrateId,
                    build_context: &BuildContext) -> ~[~str] {

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

    build_context.workcache_context.with_prep(id.install_tag(), |prep| {
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

