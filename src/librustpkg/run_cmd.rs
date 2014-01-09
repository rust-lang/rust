// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{io, os, run};
use std::io::fs;
use std::path::Path;

use context::{Context, BuildContext};
use context::{Command, BuildCmd, CleanCmd, DoCmd, InfoCmd, InstallCmd};
use context::{ListCmd, PreferCmd, TestCmd, InitCmd, UninstallCmd};
use context::{UnpreferCmd};
use crate_id::{CrateId};
use installed_packages;
use messages::{error, warn, note};
use path_util::{build_pkg_id_in_workspace, built_test_in_workspace};
use path_util::{default_workspace, dir_has_crate_file};
use perform::{build, install};
use package_source::PkgSrc;
use path_util;
use rustc::metadata::filesearch::rust_path;
use target::{WhatToBuild, Everything, Tests, MaybeCustom};
use usage;
use workspace::{each_pkg_parent_workspace, pkg_parent_workspaces, cwd_to_workspace};
use workspace::determine_destination;

/// Perform actions based on parsed command line input
pub fn run_cmd(cmd: Command,
       args: ~[~str],
       context: &Context) {
    let cwd = os::getcwd();
    match cmd {
        BuildCmd => {
            let what = WhatToBuild::new(MaybeCustom, Everything);
            run_build(args, &what, context);
        }
        CleanCmd => {
            if args.len() < 1 {
                match cwd_to_workspace() {
                    None => { usage::clean(); return }
                    // tjc: Maybe clean should clean all the packages in the
                    // current workspace, though?
                    Some((ws, crateid)) => run_clean(&ws, &crateid)
                }

            } else {
                // The package id is presumed to be the first command-line
                // argument
                let crateid = CrateId::new(args[0].clone());
                run_clean(&cwd, &crateid); // tjc: should use workspace, not cwd
            }
        }
        DoCmd => {
            if args.len() < 2 {
                return usage::do_cmd();
            }
            run_do(args[0].clone(), args[1].clone());
        }
        InfoCmd => {
            run_info();
        }
        InstallCmd => {
            if args.len() < 1 {
                run_install(None, context);
            } else {
                run_install(Some(args[0]), context);
            }
        }
        ListCmd => {
            println("Installed packages:");
            installed_packages::list_installed_packages(|pkg_id| {
                pkg_id.path.display().with_str(|s| println(s));
                true
            });
        }
        PreferCmd => {
            if args.len() < 1 {
                return usage::uninstall();
            }
            run_prefer(args[0], None);
        }
        TestCmd => {
            // Build the test executable
            let what = WhatToBuild::new(MaybeCustom, Tests);
            let maybe_id_and_workspace = run_build(args, &what, context);
            match maybe_id_and_workspace {
                Some((pkg_id, workspace)) => {
                    // Assuming it's built, run the tests
                    run_test(&pkg_id, &workspace);
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
                run_init();
            }
        }
        UninstallCmd => {
            if args.len() < 1 {
                return usage::uninstall();
            }
            run_uninstall(args[0], context);
        }
        UnpreferCmd => {
            if args.len() < 1 {
                return usage::unprefer();
            };
            run_unprefer(args[0], None);
        }
    }
}

fn run_build(args: ~[~str],
                  what: &WhatToBuild,
                  context: &Context) -> Option<(CrateId, Path)> {
    let cwd = os::getcwd();

    if args.len() < 1 {
        match cwd_to_workspace() {
            None  if dir_has_crate_file(&cwd) => {
                // FIXME (#9639): This needs to handle non-utf8 paths
                let crateid = CrateId::new(cwd.filename_str().unwrap());
                let mut pkg_src = PkgSrc::new(cwd, default_workspace(), true, crateid);
                let build_context = BuildContext::from_context(context);
                build(&mut pkg_src, what, &build_context);
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
                let build_context = BuildContext::from_context(context);
                build(&mut pkg_src, what, &build_context);
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
        each_pkg_parent_workspace(context, &crateid, |workspace| {
            debug!("found pkg {} in workspace {}, trying to build",
                   crateid.to_str(), workspace.display());
            dest_ws = determine_destination(os::getcwd(),
                                            context.use_rust_path_hack,
                                            workspace);
            let mut pkg_src = PkgSrc::new(workspace.clone(), dest_ws.clone(),
                                          false, crateid.clone());
            let build_context = BuildContext::from_context(context);
            build(&mut pkg_src, what, &build_context);
            true
        });
        // n.b. If this builds multiple packages, it only returns the workspace for
        // the last one. The whole building-multiple-packages-with-the-same-ID is weird
        // anyway and there are no tests for it, so maybe take it out
        Some((crateid, dest_ws))
    }
}


fn run_do(_cmd: &str, _pkgname: &str)  {
    // stub
    fail!("`do` not yet implemented");
}


fn run_clean(workspace: &Path, id: &CrateId)  {
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

fn run_info() {
    // stub
    fail!("info not yet implemented");
}


fn run_install(maybe_crateid_str: Option<~str>, context: &Context){
    let pkg_src = match maybe_crateid_str {
        None => match cwd_to_workspace() {
            Some((ws, crateid))                => {
                PkgSrc::new(ws.clone(), ws.clone(), false, crateid)
            },
            None if dir_has_crate_file(&os::getcwd()) => {
                // FIXME (#9639): This needs to handle non-utf8 paths
                let cwd = os::getcwd();
                let crateid = CrateId::new(cwd.filename_str().unwrap());
                PkgSrc::new(cwd, default_workspace(),true, crateid)
            }
            None  => {
                usage::install();
                return;
            }
        },
        Some(crateid_str) => {
            let crateid = CrateId::new(crateid_str);
            let workspaces = pkg_parent_workspaces(context, &crateid);
            debug!("package ID = {}, found it in {:?} workspaces",
                   crateid.to_str(), workspaces.len());
            if workspaces.is_empty() {
                // Install default workspace
                let d = default_workspace();
                PkgSrc::new(d.clone(), d, false, crateid.clone())
            } else {
                // Install all parent workspaces, and then return early
                let what = WhatToBuild::new(MaybeCustom, Everything);
                let build_context = BuildContext::from_context(context);
                for workspace in workspaces.iter() {
                    let dest = determine_destination(os::getcwd(),
                                                     context.use_rust_path_hack,
                                                     workspace);
                    let pkg_src = PkgSrc::new(workspace.clone(),
                                              dest,
                                              context.use_rust_path_hack,
                                              crateid.clone());
                    install(pkg_src, &what, &build_context);
                }
                return;
            }
        }
    };
    let what = WhatToBuild::new(MaybeCustom, Everything);
    let build_context = BuildContext::from_context(context);
    install(pkg_src, &what, &build_context);
}

fn run_prefer(_id: &str, _vers: Option<~str>)  {
    fail!("prefer not yet implemented");
}

fn run_test(crateid: &CrateId, workspace: &Path)  {
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

fn run_init() {
    fs::mkdir_recursive(&Path::new("src"), io::UserRWX);
    fs::mkdir_recursive(&Path::new("bin"), io::UserRWX);
    fs::mkdir_recursive(&Path::new("lib"), io::UserRWX);
    fs::mkdir_recursive(&Path::new("build"), io::UserRWX);
}

fn run_uninstall(crateid_str: &str, context: &Context) {
    let crateid = CrateId::new(crateid_str);
    if !installed_packages::package_is_installed(&crateid) {
        warn(format!("Package {} doesn't seem to be installed! \
                      Doing nothing.", crateid_str));
        return;
    } else {
        let rp = rust_path();
        assert!(!rp.is_empty());
        let workspaces = pkg_parent_workspaces(context, &crateid);
        for workspace in workspaces.iter() {
            path_util::uninstall_package_from(workspace, &crateid);
            note(format!("Uninstalled package {} (was installed in {})",
                      crateid.to_str(), workspace.display()));
            return
        };
        warn!("Failed to find a parent workspace")
        warn!("crateid_str: {}", crateid_str);
        warn!("crate_id: {:?}", crateid);
    }
}

fn run_unprefer(_id: &str, _vers: Option<~str>)  {
    fail!("unprefer not yet implemented");
}


