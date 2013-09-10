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
use version::Version;
use workcache_support::*;

use extra::arc::{Arc,RWArc};
use extra::workcache;
use extra::workcache::{Database, Logger, FreshnessMap};
use extra::treemap::TreeMap;

/// Convenience functions intended for calling from pkg.rs
/// p is where to put the cache file for dependencies
pub fn default_context(p: Path) -> BuildContext {
    new_default_context(new_workcache_context(&p), p)
}

pub fn new_default_context(c: workcache::Context, p: Path) -> BuildContext {
    BuildContext {
        context: Context {
            use_rust_path_hack: false,
            sysroot: p
        },
        workcache_context: c
    }
}

fn file_is_fresh(path: &str, in_hash: &str) -> bool {
    in_hash == digest_file_with_date(&Path(path))
}

fn binary_is_fresh(path: &str, in_hash: &str) -> bool {
    in_hash == digest_only_date(&Path(path))
}


pub fn new_workcache_context(p: &Path) -> workcache::Context {
    let db_file = p.push("rustpkg_db.json"); // ??? probably wrong
    debug!("Workcache database file: %s", db_file.to_str());
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
    let cx = default_context(sysroot);
    let subroot = root.clone();
    let subversion = version.clone();
    let sublib = lib.clone();
    do cx.workcache_context.with_prep(name) |prep| {
        let pkg_src = PkgSrc {
            workspace: subroot.clone(),
            start_dir: subroot.push("src").push(name),
            id: PkgId{ version: subversion.clone(), ..PkgId::new(name)},
            libs: ~[mk_crate(sublib.clone())],
            mains: ~[],
            tests: ~[],
            benchs: ~[]
        };
        pkg_src.declare_inputs(prep);
        let subcx = cx.clone();
        let subsrc = pkg_src.clone();
        do prep.exec |exec| {
            subsrc.build(exec, &subcx.clone(), ~[]);
        }
    };
}

pub fn build_exe(sysroot: Path, root: Path, name: ~str, version: Version,
                 main: Path) {
    let cx = default_context(sysroot);
    let subroot = root.clone();
    let submain = main.clone();
    do cx.workcache_context.with_prep(name) |prep| {
        let pkg_src = PkgSrc {
            workspace: subroot.clone(),
            start_dir: subroot.push("src").push(name),
            id: PkgId{ version: version.clone(), ..PkgId::new(name)},
            libs: ~[],
            mains: ~[mk_crate(submain.clone())],
            tests: ~[],
            benchs: ~[]
        };
        pkg_src.declare_inputs(prep);
        let subsrc = pkg_src.clone();
        let subcx = cx.clone();
        do prep.exec |exec| {
            subsrc.clone().build(exec, &subcx.clone(), ~[]);
        }
    }
}

pub fn install_pkg(sysroot: Path, workspace: Path, name: ~str, version: Version) {
    let cx = default_context(sysroot);
    let pkgid = PkgId{ version: version, ..PkgId::new(name)};
    cx.install(PkgSrc::new(workspace, false, pkgid));
}

fn mk_crate(p: Path) -> Crate {
    Crate { file: p, flags: ~[], cfgs: ~[] }
}
