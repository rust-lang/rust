// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustpkg utilities having to do with paths and directories

use core::path::*;
use core::{os, str};
use core::option::*;
use util::PkgId;
use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};

#[deriving(Eq)]
pub enum OutputType { Main, Lib, Bench, Test }

/// Returns the value of RUST_PATH, as a list
/// of Paths. In general this should be read from the
/// environment; for now, it's hard-wired to just be "."
pub fn rust_path() -> ~[Path] {
    ~[Path(".")]
}

static u_rwx: i32 = (S_IRUSR | S_IWUSR | S_IXUSR) as i32;

/// Creates a directory that is readable, writeable,
/// and executable by the user. Returns true iff creation
/// succeeded.
pub fn make_dir_rwx(p: &Path) -> bool {
    use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};

    os::make_dir(p, u_rwx)
}

/// Replace all occurrences of '-' in the stem part of path with '_'
/// This is because we treat rust-foo-bar-quux and rust_foo_bar_quux
/// as the same name
pub fn normalize(p: ~Path) -> ~Path {
    match p.filestem() {
        None => p,
        Some(st) => {
            let replaced = str::replace(st, "-", "_");
            if replaced != st {
                ~p.with_filestem(replaced)
            }
            else {
                p
            }
        }
    }
}

// n.b. So far this only handles local workspaces
// n.b. The next three functions ignore the package version right
// now. Should fix that.

/// True if there's a directory in <workspace> with
/// pkgid's short name
pub fn workspace_contains_package_id(pkgid: PkgId, workspace: &Path) -> bool {
    let pkgpath = workspace.push("src").push(pkgid.path.to_str());
    os::path_is_dir(&pkgpath)
}

/// Return the directory for <pkgid>'s source files in <workspace>.
/// Doesn't check that it exists.
pub fn pkgid_src_in_workspace(pkgid: PkgId, workspace: &Path) -> Path {
    let result = workspace.push("src");
    result.push(pkgid.path.to_str())
}

/// Returns the executable that would be installed for <pkgid>
/// in <workspace>
pub fn target_executable_in_workspace(pkgid: PkgId, workspace: &Path) -> Path {
    let result = workspace.push("bin");
    // should use a target-specific subdirectory
    mk_output_path(Main, pkgid.path.to_str(), result)
}


/// Returns the executable that would be installed for <pkgid>
/// in <workspace>
pub fn target_library_in_workspace(pkgid: PkgId, workspace: &Path) -> Path {
    let result = workspace.push("lib");
    mk_output_path(Lib, pkgid.path.to_str(), result)
}

/// Returns the test executable that would be installed for <pkgid>
/// in <workspace>
pub fn target_test_in_workspace(pkgid: PkgId, workspace: &Path) -> Path {
    let result = workspace.push("build");
    mk_output_path(Test, pkgid.path.to_str(), result)
}

/// Returns the bench executable that would be installed for <pkgid>
/// in <workspace>
pub fn target_bench_in_workspace(pkgid: PkgId, workspace: &Path) -> Path {
    let result = workspace.push("build");
    mk_output_path(Bench, pkgid.path.to_str(), result)
}

/// Return the directory for <pkgid>'s build artifacts in <workspace>.
/// Creates it if it doesn't exist.
pub fn build_pkg_id_in_workspace(pkgid: PkgId, workspace: &Path) -> Path {
    use conditions::bad_path::cond;

    let mut result = workspace.push("build");
    // n.b. Should actually use a target-specific
    // subdirectory of build/
    result = result.push(normalize(~pkgid.path).to_str());
    if os::path_exists(&result) || os::mkdir_recursive(&result, u_rwx) {
        result
    }
    else {
        cond.raise((result, fmt!("Could not create directory for package %s", pkgid.to_str())))
    }
}

/// Return the output file for a given directory name,
/// given whether we're building a library and whether we're building tests
pub fn mk_output_path(what: OutputType, short_name: ~str, dir: Path) -> Path {
    match what {
        Lib => dir.push(os::dll_filename(short_name)),
        _ => dir.push(fmt!("%s%s%s", short_name,
                           if what == Test { ~"test" } else { ~"" },
                           os::EXE_SUFFIX))
    }
}
