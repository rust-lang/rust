// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use context::*;
use crate::*;
use package_id::*;
use package_source::*;
use path_util::{platform_library_name, target_build_dir};
use target::*;
use version::Version;
use workspace::pkg_parent_workspaces;
use workcache_support::*;
pub use path_util::default_workspace;

pub use source_control::{safe_git_clone, git_clone_url};

use std::run;
use extra::arc::{Arc,RWArc};
use extra::workcache;
use extra::workcache::{Database, Logger, FreshnessMap};
use extra::treemap::TreeMap;

// A little sad -- duplicated from rustc::back::*
#[cfg(target_arch = "arm")]
fn cc_args() -> ~[~str] { ~[~"-marm"] }
#[cfg(target_arch = "mips")]
fn cc_args() -> ~[~str] { ~[] }
#[cfg(target_arch = "x86")]
fn cc_args() -> ~[~str] { ~[~"-m32"] }
#[cfg(target_arch = "x86_64")]
fn cc_args() -> ~[~str] { ~[~"-m64"] }

/// Convenience functions intended for calling from pkg.rs
/// p is where to put the cache file for dependencies
pub fn default_context(sysroot: Path, p: Path) -> BuildContext {
    new_default_context(new_workcache_context(&p), sysroot)
}

pub fn new_default_context(c: workcache::Context, p: Path) -> BuildContext {
    BuildContext {
        context: Context {
            cfgs: ~[],
            rustc_flags: RustcFlags::default(),
            use_rust_path_hack: false,
            sysroot: p
        },
        workcache_context: c
    }
}

fn file_is_fresh(path: &str, in_hash: &str) -> bool {
    let path = Path::new(path);
    path.exists() && in_hash == digest_file_with_date(&path)
}

fn binary_is_fresh(path: &str, in_hash: &str) -> bool {
    let path = Path::new(path);
    path.exists() && in_hash == digest_only_date(&path)
}

pub fn new_workcache_context(p: &Path) -> workcache::Context {
    let db_file = p.join("rustpkg_db.json"); // ??? probably wrong
    debug!("Workcache database file: {}", db_file.display());
    let db = RWArc::new(Database::new(db_file));
    let lg = RWArc::new(Logger::new());
    let cfg = Arc::new(TreeMap::new());
    let mut freshness: FreshnessMap = TreeMap::new();
    // Set up freshness functions for every type of dependency rustpkg
    // knows about
    freshness.insert(~"file", file_is_fresh);
    freshness.insert(~"binary", binary_is_fresh);
    workcache::Context::new_with_freshness(db, lg, cfg, Arc::new(freshness))
}

pub fn build_lib(sysroot: Path, root: Path, name: ~str, version: Version,
                 lib: Path) {
    let cx = default_context(sysroot, root.clone());
    let pkg_src = PkgSrc {
        source_workspace: root.clone(),
        build_in_destination: false,
        destination_workspace: root.clone(),
        start_dir: root.join_many(["src", name.as_slice()]),
        id: PkgId{ version: version, ..PkgId::new(name)},
        // n.b. This assumes the package only has one crate
        libs: ~[mk_crate(lib)],
        mains: ~[],
        tests: ~[],
        benchs: ~[]
    };
    pkg_src.build(&cx, ~[], []);
}

pub fn build_exe(sysroot: Path, root: Path, name: ~str, version: Version,
                 main: Path) {
    let cx = default_context(sysroot, root.clone());
    let pkg_src = PkgSrc {
        source_workspace: root.clone(),
        build_in_destination: false,
        destination_workspace: root.clone(),
        start_dir: root.join_many(["src", name.as_slice()]),
        id: PkgId{ version: version, ..PkgId::new(name)},
        libs: ~[],
        // n.b. This assumes the package only has one crate
        mains: ~[mk_crate(main)],
        tests: ~[],
        benchs: ~[]
    };

    pkg_src.build(&cx, ~[], []);
}

pub fn install_pkg(cx: &BuildContext,
                   workspace: Path,
                   name: ~str,
                   version: Version,
                   // For now, these inputs are assumed to be inputs to each of the crates
                   more_inputs: ~[(~str, Path)]) { // pairs of Kind and Path
    let pkgid = PkgId{ version: version, ..PkgId::new(name)};
    cx.install(PkgSrc::new(workspace.clone(), workspace, false, pkgid),
               &WhatToBuild{ build_type: Inferred,
                             inputs_to_discover: more_inputs,
                             sources: Everything });
}

/// Builds an arbitrary library whose short name is `output`,
/// by invoking `tool` with arguments `args` plus "-o %s", where %s
/// is the platform-specific library name for `output`.
/// Returns that platform-specific name.
pub fn build_library_in_workspace(exec: &mut workcache::Exec,
                                  context: &mut Context,
                                  package_name: &str,
                                  tool: &str,
                                  flags: &[~str],
                                  paths: &[~str],
                                  output: &str) -> ~str {
    use command_failed = conditions::command_failed::cond;

    let workspace = my_workspace(context, package_name);
    let workspace_build_dir = target_build_dir(&workspace);
    let out_name = workspace_build_dir.join_many([package_name.to_str(),
                                                  platform_library_name(output)]);
    // make paths absolute
    let pkgid = PkgId::new(package_name);
    let absolute_paths = paths.map(|s| {
            let whatever = workspace.join_many([~"src",
                                pkgid.to_str(),
                                s.to_owned()]);
            whatever.as_str().unwrap().to_owned()
        });

    let cc_args = cc_args();

    let all_args = flags + absolute_paths + cc_args +
         ~[~"-o", out_name.as_str().unwrap().to_owned()];
    let exit_process = run::process_status(tool, all_args);
    if exit_process.success() {
        let out_name_str = out_name.as_str().unwrap().to_owned();
        exec.discover_output("binary",
                             out_name_str,
                             digest_only_date(&out_name));
        context.add_library_path(out_name.dir_path());
        out_name_str
    } else {
        command_failed.raise((tool.to_owned(), all_args, exit_process))
    }
}

pub fn my_workspace(context: &Context, package_name: &str) -> Path {
    use bad_pkg_id     = conditions::bad_pkg_id::cond;

    // (this assumes no particular version is requested)
    let pkgid = PkgId::new(package_name);
    let workspaces = pkg_parent_workspaces(context, &pkgid);
    if workspaces.is_empty() {
        bad_pkg_id.raise((Path::new(package_name), package_name.to_owned()));
    }
    workspaces[0]
}

fn mk_crate(p: Path) -> Crate {
    Crate { file: p, flags: ~[], cfgs: ~[] }
}
