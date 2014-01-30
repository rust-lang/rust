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

#[allow(dead_code)];

pub use target::{OutputType, Main, Lib, Test, Bench, Target, Build, Install};
pub use version::{Version, split_version_general};
pub use rustc::metadata::filesearch::rust_path;

use std::libc;
use std::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
use std::os;
use std::io;
use std::io::fs;
use extra::hex::ToHex;
use syntax::crateid::CrateId;
use rustc::util::sha2::{Digest, Sha256};
use rustc::metadata::filesearch::{libdir, relative_target_lib_path};
use rustc::driver::driver::host_triple;
use messages::*;

pub fn default_workspace() -> Path {
    let p = rust_path();
    if p.is_empty() {
        fail!("Empty RUST_PATH");
    }
    let result = p[0];
    if !result.is_dir() {
        fs::mkdir_recursive(&result, io::UserRWX);
    }
    result
}

pub fn in_rust_path(p: &Path) -> bool {
    rust_path().contains(p)
}

pub static U_RWX: i32 = (S_IRUSR | S_IWUSR | S_IXUSR) as i32;

/// Creates a directory that is readable, writeable,
/// and executable by the user. Returns true iff creation
/// succeeded.
pub fn make_dir_rwx(p: &Path) -> bool {
    io::result(|| fs::mkdir(p, io::UserRWX)).is_ok()
}

pub fn make_dir_rwx_recursive(p: &Path) -> bool {
    io::result(|| fs::mkdir_recursive(p, io::UserRWX)).is_ok()
}

// n.b. The next three functions ignore the package version right
// now. Should fix that.

/// True if there's a directory in <workspace> with
/// crateid's short name
pub fn workspace_contains_crate_id(crateid: &CrateId, workspace: &Path) -> bool {
    workspace_contains_crate_id_(crateid, workspace, |p| p.join("src")).is_some()
}

pub fn workspace_contains_crate_id_(crateid: &CrateId, workspace: &Path,
// Returns the directory it was actually found in
             workspace_to_src_dir: |&Path| -> Path) -> Option<Path> {
    if !workspace.is_dir() {
        return None;
    }

    let src_dir = workspace_to_src_dir(workspace);
    if !src_dir.is_dir() { return None }

    let mut found = None;
    for p in fs::walk_dir(&src_dir) {
        if p.is_dir() {
            if p == src_dir.join(crateid.path.as_slice()) || {
                let pf = p.filename_str();
                pf.iter().any(|&g| {
                    match split_version_general(g, '-') {
                        None => false,
                        Some((ref might_match, ref vers)) => {
                            *might_match == crateid.name
                                && (crateid.version == *vers || crateid.version == None)
                        }
                    }
                })
            } {
                found = Some(p.clone());
            }

        }
    }

    if found.is_some() {
        debug!("Found {} in {}", crateid.to_str(), workspace.display());
    } else {
        debug!("Didn't find {} in {}", crateid.to_str(), workspace.display());
    }
    found
}

/// Return the target-specific build subdirectory, pushed onto `base`;
/// doesn't check that it exists or create it
pub fn target_build_dir(workspace: &Path) -> Path {
    let mut dir = workspace.join("build");
    dir.push(host_triple());
    dir
}

/// Return the target-specific lib subdirectory, pushed onto `base`;
/// doesn't check that it exists or create it
fn target_lib_dir(workspace: &Path) -> Path {
    let mut dir = workspace.join(libdir());
    dir.push(host_triple());
    dir
}

/// Return the bin subdirectory, pushed onto `base`;
/// doesn't check that it exists or create it
/// note: this isn't target-specific
fn target_bin_dir(workspace: &Path) -> Path {
    workspace.join("bin")
}

/// Figure out what the executable name for <crateid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_executable_in_workspace(crateid: &CrateId, workspace: &Path) -> Option<Path> {
    let mut result = target_build_dir(workspace);
    result = mk_output_path(Main, Build, crateid, result);
    debug!("built_executable_in_workspace: checking whether {} exists",
           result.display());
    if result.exists() {
        Some(result)
    }
    else {
        debug!("built_executable_in_workspace: {} does not exist", result.display());
        None
    }
}

/// Figure out what the test name for <crateid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_test_in_workspace(crateid: &CrateId, workspace: &Path) -> Option<Path> {
    output_in_workspace(crateid, workspace, Test)
}

/// Figure out what the test name for <crateid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_bench_in_workspace(crateid: &CrateId, workspace: &Path) -> Option<Path> {
    output_in_workspace(crateid, workspace, Bench)
}

fn output_in_workspace(crateid: &CrateId, workspace: &Path, what: OutputType) -> Option<Path> {
    let mut result = target_build_dir(workspace);
    // should use a target-specific subdirectory
    result = mk_output_path(what, Build, crateid, result);
    debug!("output_in_workspace: checking whether {} exists",
           result.display());
    if result.exists() {
        Some(result)
    }
    else {
        error!("output_in_workspace: {} does not exist", result.display());
        None
    }
}

/// Figure out what the library name for <crateid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_library_in_workspace(crateid: &CrateId, workspace: &Path) -> Option<Path> {
    library_in_workspace(crateid, Build, workspace)
}

/// Does the actual searching stuff
pub fn installed_library_in_workspace(crate_id: &CrateId, workspace: &Path) -> Option<Path> {
    // This could break once we're handling multiple versions better -- I should add a test for it
    let path = Path::new(crate_id.path.as_slice());
    match path.filename_str() {
        None => None,
        Some(_short_name) => library_in_workspace(crate_id, Install, workspace)
    }
}

/// `workspace` is used to figure out the directory to search.
/// `name` is taken as the link name of the library.
pub fn library_in_workspace(crate_id: &CrateId, where: Target, workspace: &Path) -> Option<Path> {
    debug!("library_in_workspace: checking whether a library named {} exists",
           crate_id.name);

    let dir_to_search = match where {
        Build => target_build_dir(workspace).join(crate_id.path.as_slice()),
        Install => target_lib_dir(workspace)
    };

    library_in(crate_id, &dir_to_search)
}

pub fn system_library(sysroot: &Path, crate_id: &CrateId) -> Option<Path> {
    library_in(crate_id, &sysroot.join(relative_target_lib_path(host_triple())))
}

fn library_in(crate_id: &CrateId, dir_to_search: &Path) -> Option<Path> {
    let mut hasher = Sha256::new();
    hasher.reset();
    hasher.input_str(crate_id.to_str());
    let hash = hasher.result_bytes().to_hex();
    let hash = hash.slice_chars(0, 8);

    let lib_name = format!("{}-{}-{}", crate_id.name, hash, crate_id.version_or_default());
    let filenames = [
        format!("{}{}.{}", "lib", lib_name, "rlib"),
        format!("{}{}{}", os::consts::DLL_PREFIX, lib_name, os::consts::DLL_SUFFIX),
    ];

    for filename in filenames.iter() {
        debug!("filename = {}", filename.as_slice());
        let path = dir_to_search.join(filename.as_slice());
        if path.exists() {
            debug!("found: {}", path.display());
            return Some(path);
        }
    }
    debug!("warning: library_in_workspace didn't find a library in {} for {}",
           dir_to_search.display(), crate_id.to_str());
    return None;
}

/// Returns the executable that would be installed for <crateid>
/// in <workspace>
/// As a side effect, creates the bin-dir if it doesn't exist
pub fn target_executable_in_workspace(crateid: &CrateId, workspace: &Path) -> Path {
    target_file_in_workspace(crateid, workspace, Main, Install)
}


/// Returns the executable that would be installed for <crateid>
/// in <workspace>
/// As a side effect, creates the lib-dir if it doesn't exist
pub fn target_library_in_workspace(crateid: &CrateId, workspace: &Path) -> Path {
    use conditions::bad_path::cond;
    if !workspace.is_dir() {
        cond.raise(((*workspace).clone(),
                    format!("Workspace supplied to target_library_in_workspace \
                             is not a directory! {}", workspace.display())));
    }
    target_file_in_workspace(crateid, workspace, Lib, Install)
}

/// Returns the test executable that would be installed for <crateid>
/// in <workspace>
/// note that we *don't* install test executables, so this is just for unit testing
pub fn target_test_in_workspace(crateid: &CrateId, workspace: &Path) -> Path {
    target_file_in_workspace(crateid, workspace, Test, Install)
}

/// Returns the bench executable that would be installed for <crateid>
/// in <workspace>
/// note that we *don't* install bench executables, so this is just for unit testing
pub fn target_bench_in_workspace(crateid: &CrateId, workspace: &Path) -> Path {
    target_file_in_workspace(crateid, workspace, Bench, Install)
}


/// Returns the path that crateid `crateid` would have if placed `where`
/// in `workspace`
fn target_file_in_workspace(crateid: &CrateId, workspace: &Path,
                            what: OutputType, where: Target) -> Path {
    use conditions::bad_path::cond;

    let subdir = match what {
        Lib => "lib", Main | Test | Bench => "bin"
    };
    // Artifacts in the build directory live in a package-ID-specific subdirectory,
    // but installed ones don't.
    let result = match (where, what) {
                (Build, _)      => target_build_dir(workspace).join(crateid.path.as_slice()),
                (Install, Lib)  => target_lib_dir(workspace),
                (Install, _)    => target_bin_dir(workspace)
    };
    if io::result(|| fs::mkdir_recursive(&result, io::UserRWX)).is_err() {
        cond.raise((result.clone(), format!("target_file_in_workspace couldn't \
            create the {} dir (crateid={}, workspace={}, what={:?}, where={:?}",
            subdir, crateid.to_str(), workspace.display(), what, where)));
    }
    mk_output_path(what, where, crateid, result)
}

/// Return the directory for <crateid>'s build artifacts in <workspace>.
/// Creates it if it doesn't exist.
pub fn build_pkg_id_in_workspace(crateid: &CrateId, workspace: &Path) -> Path {
    let mut result = target_build_dir(workspace);
    result.push(crateid.path.as_slice());
    debug!("Creating build dir {} for package id {}", result.display(),
           crateid.to_str());
    fs::mkdir_recursive(&result, io::UserRWX);
    return result;
}

/// Return the output file for a given directory name,
/// given whether we're building a library and whether we're building tests
pub fn mk_output_path(what: OutputType, where: Target,
                      crate_id: &CrateId, workspace: Path) -> Path {
    let short_name_with_version = crate_id.short_name_with_version();
    // Not local_path.dir_path()! For package foo/bar/blat/, we want
    // the executable blat-0.5 to live under blat/
    let dir = match where {
        // If we're installing, it just goes under <workspace>...
        Install => workspace,
        // and if we're just building, it goes in a package-specific subdir
        Build => workspace.join(crate_id.path.as_slice())
    };
    debug!("[{:?}:{:?}] mk_output_path: name = {}, path = {}", what, where,
           if what == Lib { short_name_with_version.clone() } else { crate_id.name.clone() },
           dir.display());
    let mut output_path = match what {
        // this code is duplicated from elsewhere; fix this
        Lib => dir.join(os::dll_filename(short_name_with_version)),
        // executable names *aren't* versioned
        _ => dir.join(format!("{}{}{}", crate_id.name,
                           match what {
                               Test => "test",
                               Bench => "bench",
                               _     => ""
                           },
                           os::consts::EXE_SUFFIX))
    };
    if !output_path.is_absolute() {
        output_path = os::getcwd().join(&output_path);
    }
    debug!("mk_output_path: returning {}", output_path.display());
    output_path
}

/// Removes files for the package `crateid`, assuming it's installed in workspace `workspace`
pub fn uninstall_package_from(workspace: &Path, crateid: &CrateId) {
    let mut did_something = false;
    let installed_bin = target_executable_in_workspace(crateid, workspace);
    if installed_bin.exists() {
        fs::unlink(&installed_bin);
        did_something = true;
    }
    let installed_lib = target_library_in_workspace(crateid, workspace);
    if installed_lib.exists() {
        fs::unlink(&installed_lib);
        did_something = true;
    }
    if !did_something {
        warn(format!("Warning: there don't seem to be any files for {} installed in {}",
             crateid.to_str(), workspace.display()));
    }

}

pub fn dir_has_crate_file(dir: &Path) -> bool {
    dir_has_file(dir, "lib.rs") || dir_has_file(dir, "main.rs")
        || dir_has_file(dir, "test.rs") || dir_has_file(dir, "bench.rs")
}

fn dir_has_file(dir: &Path, file: &str) -> bool {
    assert!(dir.is_absolute());
    dir.join(file).exists()
}

pub fn find_dir_using_rust_path_hack(p: &CrateId) -> Option<Path> {
    let rp = rust_path();
    let path = Path::new(p.path.as_slice());
    for dir in rp.iter() {
        // Require that the parent directory match the package ID
        // Note that this only matches if the package ID being searched for
        // has a name that's a single component
        if dir.ends_with_path(&path) || dir.ends_with_path(&versionize(p.path, &p.version)) {
            debug!("In find_dir_using_rust_path_hack: checking dir {}", dir.display());
            if dir_has_crate_file(dir) {
                debug!("Did find id {} in dir {}", p.to_str(), dir.display());
                return Some(dir.clone());
            }
        }
        debug!("Didn't find id {} in dir {}", p.to_str(), dir.display())
    }
    None
}

/// True if the user set RUST_PATH to something non-empty --
/// as opposed to the default paths that rustpkg adds automatically
pub fn user_set_rust_path() -> bool {
    match os::getenv("RUST_PATH") {
        None | Some(~"") => false,
        Some(_)         => true
    }
}

/// Append the version string onto the end of the path's filename
pub fn versionize(p: &str, v: &Version) -> Path {
    let p = Path::new(p);
    let q = p.filename().expect("path is a directory");
    let mut q = q.to_owned();
    q.push('-' as u8);
    let vs = match v { &Some(ref s) => s.to_owned(), &None => ~"0.0" };
    q.push_all(vs.as_bytes());
    p.with_filename(q)
}

#[cfg(target_os = "win32")]
pub fn chmod_read_only(p: &Path) -> bool {
    unsafe {
        p.with_c_str(|src_buf| libc::chmod(src_buf, S_IRUSR as libc::c_int) == 0 as libc::c_int)
    }
}

#[cfg(not(target_os = "win32"))]
pub fn chmod_read_only(p: &Path) -> bool {
    unsafe {
        p.with_c_str(|src_buf| libc::chmod(src_buf, S_IRUSR as libc::mode_t) == 0 as libc::c_int)
    }
}

pub fn platform_library_name(s: &str) -> ~str {
    format!("{}{}{}", os::consts::DLL_PREFIX, s, os::consts::DLL_SUFFIX)
}
