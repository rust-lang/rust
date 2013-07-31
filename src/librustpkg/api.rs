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

use std::os;
use std::hashmap::*;

/// Convenience functions intended for calling from pkg.rs

fn default_ctxt(p: @Path) -> Ctx {
    Ctx { sysroot_opt: Some(p), json: false, dep_cache: @mut HashMap::new() }
}

pub fn build_lib(sysroot: @Path, root: Path, name: ~str, version: Version,
                 lib: Path) {

    let pkg_src = PkgSrc {
        root: root,
        id: PkgId{ version: version, ..PkgId::new(name)},
        libs: ~[mk_crate(lib)],
        mains: ~[],
        tests: ~[],
        benchs: ~[]
    };
    pkg_src.build(&default_ctxt(sysroot), ~[]);
}

pub fn build_exe(sysroot: @Path, root: Path, name: ~str, version: Version, main: Path) {
    let pkg_src = PkgSrc {
        root: root,
        id: PkgId{ version: version, ..PkgId::new(name)},
        libs: ~[],
        mains: ~[mk_crate(main)],
        tests: ~[],
        benchs: ~[]
    };
    pkg_src.build(&default_ctxt(sysroot), ~[]);

}

pub fn install_lib(sysroot: @Path,
                   workspace: Path,
                   name: ~str,
                   lib_path: Path,
                   version: Version) {
    debug!("self_exe: %?", os::self_exe_path());
    debug!("sysroot = %s", sysroot.to_str());
    debug!("workspace = %s", workspace.to_str());
    // make a PkgSrc
    let pkg_id = PkgId{ version: version, ..PkgId::new(name)};
    let pkg_src = PkgSrc {
        root: workspace.clone(),
        id: pkg_id.clone(),
        libs: ~[mk_crate(lib_path)],
        mains: ~[],
        tests: ~[],
        benchs: ~[]
    };
    let cx = default_ctxt(sysroot);
    pkg_src.build(&cx, ~[]);
    cx.install_no_build(&workspace, &pkg_id);
}

pub fn install_exe(sysroot: @Path, workspace: Path, name: ~str, version: Version) {
    default_ctxt(sysroot).install(&workspace, &PkgId{ version: version,
                                            ..PkgId::new(name)});

}

fn mk_crate(p: Path) -> Crate {
    Crate { file: p, flags: ~[], cfgs: ~[] }
}
