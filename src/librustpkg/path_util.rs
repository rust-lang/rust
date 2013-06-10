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

use core::prelude::*;
pub use package_path::{RemotePath, LocalPath};
pub use package_id::PkgId;
pub use target::{OutputType, Main, Lib, Test, Bench, Target, Build, Install};
use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
use core::os::mkdir_recursive;
use core::os;
use core::iterator::IteratorUtil;

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

// n.b. The next three functions ignore the package version right
// now. Should fix that.

/// True if there's a directory in <workspace> with
/// pkgid's short name
pub fn workspace_contains_package_id(pkgid: &PkgId, workspace: &Path) -> bool {
    let pkgpath = workspace.push("src").push(pkgid.remote_path.to_str());
    os::path_is_dir(&pkgpath)
}

/// Returns a list of possible directories
/// for <pkgid>'s source files in <workspace>.
/// Doesn't check that any of them exist.
/// (for example, try both with and without the version)
pub fn pkgid_src_in_workspace(pkgid: &PkgId, workspace: &Path) -> ~[Path] {
    let mut results = ~[];
    let result = workspace.push("src").push(fmt!("%s-%s",
                     pkgid.local_path.to_str(), pkgid.version.to_str()));
    results.push(result);
    results.push(workspace.push("src").push_rel(&*pkgid.remote_path));
    results
}

/// Returns a src for pkgid that does exist -- None if none of them do
pub fn first_pkgid_src_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
    let rs = pkgid_src_in_workspace(pkgid, workspace);
    for rs.each |p| {
        if os::path_exists(p) {
            return Some(copy *p);
        }
    }
    None
}

/// Figure out what the executable name for <pkgid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_executable_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
    let mut result = workspace.push("build");
    // should use a target-specific subdirectory
    result = mk_output_path(Main, Build, pkgid, &result);
    debug!("built_executable_in_workspace: checking whether %s exists",
           result.to_str());
    if os::path_exists(&result) {
        Some(result)
    }
    else {
        // This is not an error, but it's worth logging it
        error!(fmt!("built_executable_in_workspace: %s does not exist", result.to_str()));
        None
    }
}

/// Figure out what the test name for <pkgid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_test_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
    output_in_workspace(pkgid, workspace, Test)
}

/// Figure out what the test name for <pkgid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_bench_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
    output_in_workspace(pkgid, workspace, Bench)
}

fn output_in_workspace(pkgid: &PkgId, workspace: &Path, what: OutputType) -> Option<Path> {
    let mut result = workspace.push("build");
    // should use a target-specific subdirectory
    result = mk_output_path(what, Build, pkgid, &result);
    debug!("output_in_workspace: checking whether %s exists",
           result.to_str());
    if os::path_exists(&result) {
        Some(result)
    }
    else {
        error!(fmt!("output_in_workspace: %s does not exist", result.to_str()));
        None
    }
}

/// Figure out what the library name for <pkgid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_library_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
                        // passing in local_path here sounds fishy
    library_in_workspace(pkgid.local_path.to_str(), pkgid.short_name, Build,
                         workspace, "build")
}

/// Does the actual searching stuff
pub fn installed_library_in_workspace(short_name: &str, workspace: &Path) -> Option<Path> {
    library_in_workspace(short_name, short_name, Install, workspace, "lib")
}


/// This doesn't take a PkgId, so we can use it for `extern mod` inference, where we
/// don't know the entire package ID.
/// `full_name` is used to figure out the directory to search.
/// `short_name` is taken as the link name of the library.
fn library_in_workspace(full_name: &str, short_name: &str, where: Target,
                        workspace: &Path, prefix: &str) -> Option<Path> {
    debug!("library_in_workspace: checking whether a library named %s exists",
           short_name);

    // We don't know what the hash is, so we have to search through the directory
    // contents

    let dir_to_search = match where {
        Build => workspace.push(prefix).push(full_name),
        Install => workspace.push(prefix)
    };
    debug!("Listing directory %s", dir_to_search.to_str());
    let dir_contents = os::list_dir(&dir_to_search);
    debug!("dir has %? entries", dir_contents.len());

    let lib_prefix = fmt!("%s%s", os::consts::DLL_PREFIX, short_name);
    let lib_filetype = os::consts::DLL_SUFFIX;

    debug!("lib_prefix = %s and lib_filetype = %s", lib_prefix, lib_filetype);

    let mut result_filename = None;
    for dir_contents.each |&p| {
        let mut which = 0;
        let mut hash = None;
        let p_path = Path(p);
        let extension = p_path.filetype();
        debug!("p = %s, p's extension is %?", p.to_str(), extension);
        match extension {
            Some(ref s) if lib_filetype == *s => (),
            _ => loop
        }
        // Find a filename that matches the pattern: (lib_prefix)-hash-(version)(lib_suffix)
        // and remember what the hash was
        let f_name = match p_path.filename() {
            Some(s) => s, None => loop
        };
        for f_name.split_iter('-').advance |piece| {
            debug!("a piece = %s", piece);
            if which == 0 && piece != lib_prefix {
                break;
            }
            else if which == 0 {
                which += 1;
            }
            else if which == 1 {
                hash = Some(piece.to_owned());
                break;
            }
            else {
                // something went wrong
                hash = None;
                break;
            }
        }
        if hash.is_some() {
            result_filename = Some(p_path);
            break;
        }
    }

    // Return the filename that matches, which we now know exists
    // (if result_filename != None)
    match result_filename {
        None => None,
        Some(result_filename) => {
            let absolute_path = dir_to_search.push_rel(&result_filename);
            debug!("result_filename = %s", absolute_path.to_str());
            Some(absolute_path)
        }
    }
}

/// Returns the executable that would be installed for <pkgid>
/// in <workspace>
/// As a side effect, creates the bin-dir if it doesn't exist
pub fn target_executable_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    target_file_in_workspace(pkgid, workspace, Main, Install)
}


/// Returns the installed path for <built_library> in <workspace>
/// As a side effect, creates the lib-dir if it doesn't exist
pub fn target_library_in_workspace(workspace: &Path,
                                   built_library: &Path) -> Path {
    use conditions::bad_path::cond;
    let result = workspace.push("lib");
    if !os::path_exists(&result) && !mkdir_recursive(&result, u_rwx) {
        cond.raise((copy result, ~"I couldn't create the library directory"));
    }
    result.push(built_library.filename().expect(fmt!("I don't know how to treat %s as a library",
                                                   built_library.to_str())))
}

/// Returns the test executable that would be installed for <pkgid>
/// in <workspace>
/// note that we *don't* install test executables, so this is just for unit testing
pub fn target_test_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    target_file_in_workspace(pkgid, workspace, Test, Install)
}

/// Returns the bench executable that would be installed for <pkgid>
/// in <workspace>
/// note that we *don't* install bench executables, so this is just for unit testing
pub fn target_bench_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    target_file_in_workspace(pkgid, workspace, Bench, Install)
}


/// Returns the path that pkgid `pkgid` would have if placed `where`
/// in `workspace`
fn target_file_in_workspace(pkgid: &PkgId, workspace: &Path,
                            what: OutputType, where: Target) -> Path {
    use conditions::bad_path::cond;

    let subdir = match what {
        Lib => "lib", Main | Test | Bench => "bin"
    };
    let result = workspace.push(subdir);
    if !os::path_exists(&result) && !mkdir_recursive(&result, u_rwx) {
        cond.raise((copy result, fmt!("I couldn't create the %s dir", subdir)));
    }
    mk_output_path(what, where, pkgid, &result)
}

/// Return the directory for <pkgid>'s build artifacts in <workspace>.
/// Creates it if it doesn't exist.
pub fn build_pkg_id_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    use conditions::bad_path::cond;

    let mut result = workspace.push("build");
    // n.b. Should actually use a target-specific
    // subdirectory of build/
    result = result.push_rel(&*pkgid.local_path);
    if os::path_exists(&result) || os::mkdir_recursive(&result, u_rwx) {
        result
    }
    else {
        cond.raise((result, fmt!("Could not create directory for package %s", pkgid.to_str())))
    }
}

/// Return the output file for a given directory name,
/// given whether we're building a library and whether we're building tests
pub fn mk_output_path(what: OutputType, where: Target,
                      pkg_id: &PkgId, workspace: &Path) -> Path {
    let short_name_with_version = pkg_id.short_name_with_version();
    // Not local_path.dir_path()! For package foo/bar/blat/, we want
    // the executable blat-0.5 to live under blat/
    let dir = match where {
        // If we're installing, it just goes under <workspace>...
        Install => copy *workspace, // bad copy, but I just couldn't make the borrow checker happy
        // and if we're just building, it goes in a package-specific subdir
        Build => workspace.push_rel(&*pkg_id.local_path)
    };
    debug!("[%?:%?] mk_output_path: short_name = %s, path = %s", what, where,
           if what == Lib { copy short_name_with_version } else { copy pkg_id.short_name },
           dir.to_str());
    let mut output_path = match what {
        // this code is duplicated from elsewhere; fix this
        Lib => dir.push(os::dll_filename(short_name_with_version)),
        // executable names *aren't* versioned
        _ => dir.push(fmt!("%s%s%s", copy pkg_id.short_name,
                           match what {
                               Test => "test",
                               Bench => "bench",
                               _     => ""
                           }
                           os::EXE_SUFFIX))
    };
    if !output_path.is_absolute() {
        output_path = os::getcwd().push_rel(&output_path).normalize();
    }
    debug!("mk_output_path: returning %s", output_path.to_str());
    output_path
}
