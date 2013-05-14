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

use util::PkgId;
use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
use core::os::mkdir_recursive;

#[deriving(Eq)]
pub enum OutputType { Main, Lib, Bench, Test }

/// Returns the value of RUST_PATH, as a list
/// of Paths. In general this should be read from the
/// environment; for now, it's hard-wired to just be "."
pub fn rust_path() -> ~[Path] {
    ~[Path(".")]
}

pub static u_rwx: i32 = (S_IRUSR | S_IWUSR | S_IXUSR) as i32;

/// Creates a directory that is readable, writeable,
/// and executable by the user. Returns true iff creation
/// succeeded.
pub fn make_dir_rwx(p: &Path) -> bool { os::make_dir(p, u_rwx) }

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
pub fn workspace_contains_package_id(pkgid: &PkgId, workspace: &Path) -> bool {
    let pkgpath = workspace.push("src").push(pkgid.path.to_str());
    os::path_is_dir(&pkgpath)
}

/// Return the directory for <pkgid>'s source files in <workspace>.
/// Doesn't check that it exists.
pub fn pkgid_src_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    let result = workspace.push("src");
    result.push(pkgid.path.to_str())
}

/// Figure out what the executable name for <pkgid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_executable_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
    let mut result = workspace.push("build");
    result = result.push_rel(&pkgid.path);
    // should use a target-specific subdirectory
    result = mk_output_path(Main, fmt!("%s-%s", pkgid.path.to_str(), pkgid.version.to_str()),
                                       result);
    debug!("built_executable_in_workspace: checking whether %s exists",
           result.to_str());
    if os::path_exists(&result) {
        Some(result)
    }
    else {
        None
    }
}

/// Figure out what the library name for <pkgid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_library_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
    let mut result = workspace.push("build");
    result = result.push_rel(&pkgid.path);
    // should use a target-specific subdirectory
    result = mk_output_path(Lib, pkgid.path.to_str(), result);
    debug!("built_library_in_workspace: checking whether %s exists",
           result.to_str());

    // We don't know what the hash is, so we have to search through the directory
    // contents
    let dir_contents = os::list_dir(&result.pop());
    debug!("dir has %? entries", dir_contents.len());

    // n.b. This code assumes the pkgid's path only has one element
    let lib_prefix = fmt!("%s%s", os::consts::DLL_PREFIX, pkgid.path.to_str());
    let lib_filetype = fmt!("%s%s", pkgid.version.to_str(), os::consts::DLL_SUFFIX);

    debug!("lib_prefix = %s and lib_filetype = %s", lib_prefix, lib_filetype);

    let mut result_filename = None;
    for dir_contents.each |&p| {
        let mut which = 0;
        let mut hash = None;
        // Find a filename that matches the pattern: (lib_prefix)-hash-(version)(lib_suffix)
        // and remember what the hash was
        for p.each_split_char('-') |piece| {
            debug!("a piece = %s", piece);
            if which == 0 && piece != lib_prefix {
                break;
            }
            else if which == 0 {
                which += 1;
            }
            else if which == 1 {
                hash = Some(piece.to_owned());
                which += 1;
            }
            else if which == 2 && piece != lib_filetype {
                hash = None;
                break;
            }
            else if which == 2 {
                break;
            }
            else {
                // something went wrong
                hash = None;
                break;
            }
        }
        if hash.is_some() {
            result_filename = Some(p);
            break;
        }
    }

    // Return the filename that matches, which we now know exists
    // (if result_filename != None)
    debug!("result_filename = %?", result_filename);
    match result_filename {
        None => None,
        Some(result_filename) => {
            let result_filename = result.with_filename(result_filename);
            debug!("result_filename = %s", result_filename.to_str());
            Some(result_filename)
        }
    }
}

/// Returns the executable that would be installed for <pkgid>
/// in <workspace>
/// As a side effect, creates the bin-dir if it doesn't exist
pub fn target_executable_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    target_file_in_workspace(pkgid, workspace, Main)
}


/// Returns the executable that would be installed for <pkgid>
/// in <workspace>
/// As a side effect, creates the bin-dir if it doesn't exist
pub fn target_library_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    target_file_in_workspace(pkgid, workspace, Lib)
}

/// Returns the test executable that would be installed for <pkgid>
/// in <workspace>
pub fn target_test_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    target_file_in_workspace(pkgid, workspace, Test)
}

/// Returns the bench executable that would be installed for <pkgid>
/// in <workspace>
pub fn target_bench_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    target_file_in_workspace(pkgid, workspace, Bench)
}

fn target_file_in_workspace(pkgid: &PkgId, workspace: &Path,
                            what: OutputType) -> Path {
    use conditions::bad_path::cond;

    let (subdir, create_dir) = match what {
        Main => ("bin", true), Lib => ("lib", true), Test | Bench => ("build", false)
    };
    let result = workspace.push(subdir);
    if create_dir {
        if !os::path_exists(&result) && !mkdir_recursive(&result, u_rwx) {
            cond.raise((copy result,
                        fmt!("I couldn't create the %s dir", subdir)));
        }
    }
    mk_output_path(what, pkgid.path.to_str(), result)

}

/// Return the directory for <pkgid>'s build artifacts in <workspace>.
/// Creates it if it doesn't exist.
pub fn build_pkg_id_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    use conditions::bad_path::cond;

    let mut result = workspace.push("build");
    // n.b. Should actually use a target-specific
    // subdirectory of build/
    result = result.push(normalize(~copy pkgid.path).to_str());
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
                           match what {
                               Test => "test",
                               Bench => "bench",
                               _     => ""
                           }
                           os::EXE_SUFFIX))
    }
}
