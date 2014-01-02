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

pub use crate_id::CrateId;
pub use target::{OutputType, Main, Lib, Test, Bench, Target, Build, Install};
pub use version::{Version, ExactRevision, NoVersion, split_version, split_version_general,
    try_parsing_version};
pub use rustc::metadata::filesearch::rust_path;
use rustc::metadata::filesearch::libdir;
use rustc::driver::driver::host_triple;

use std::libc;
use std::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
use std::os;
use std::io;
use std::io::fs;
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
            if p == src_dir.join(&crateid.path) || {
                let pf = p.filename_str();
                pf.iter().any(|&g| {
                    match split_version_general(g, '-') {
                        None => false,
                        Some((ref might_match, ref vers)) => {
                            *might_match == crateid.short_name
                                && (crateid.version == *vers || crateid.version == NoVersion)
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
    library_in_workspace(&crateid.path, crateid.short_name, Build, workspace, "build",
                         &crateid.version)
}

/// Does the actual searching stuff
pub fn installed_library_in_workspace(pkg_path: &Path, workspace: &Path) -> Option<Path> {
    // This could break once we're handling multiple versions better -- I should add a test for it
    // FIXME (#9639): This needs to handle non-utf8 paths
    match pkg_path.filename_str() {
        None => None,
        Some(short_name) => library_in_workspace(pkg_path,
                                                 short_name,
                                                 Install,
                                                 workspace,
                                                 libdir(),
                                                 &NoVersion)
    }
}

/// `workspace` is used to figure out the directory to search.
/// `short_name` is taken as the link name of the library.
pub fn library_in_workspace(path: &Path, short_name: &str, where: Target,
                        workspace: &Path, prefix: &str, version: &Version) -> Option<Path> {
    debug!("library_in_workspace: checking whether a library named {} exists",
           short_name);

    // We don't know what the hash is, so we have to search through the directory
    // contents

    debug!("short_name = {} where = {:?} workspace = {} \
            prefix = {}", short_name, where, workspace.display(), prefix);

    let dir_to_search = match where {
        Build => target_build_dir(workspace).join(path),
        Install => target_lib_dir(workspace)
    };

    library_in(short_name, version, &dir_to_search)
}

// rustc doesn't use target-specific subdirectories
pub fn system_library(sysroot: &Path, crate_id: &str) -> Option<Path> {
    let (lib_name, version) = split_crate_id(crate_id);
    library_in(lib_name, &version, &sysroot.join(libdir()))
}

fn library_in(short_name: &str, version: &Version, dir_to_search: &Path) -> Option<Path> {
    debug!("Listing directory {}", dir_to_search.display());
    let dir_contents = {
        let _guard = io::ignore_io_error();
        fs::readdir(dir_to_search)
    };
    debug!("dir has {:?} entries", dir_contents.len());

    let lib_prefix = format!("{}{}", os::consts::DLL_PREFIX, short_name);
    let lib_filetype = os::consts::DLL_EXTENSION;

    debug!("lib_prefix = {} and lib_filetype = {}", lib_prefix, lib_filetype);

    // Find a filename that matches the pattern:
    // (lib_prefix)-hash-(version)(lib_suffix)
    let mut libraries = dir_contents.iter().filter(|p| {
        let extension = p.extension_str();
        debug!("p = {}, p's extension is {:?}", p.display(), extension);
        match extension {
            None => false,
            Some(ref s) => lib_filetype == *s
        }
    });

    let mut result_filename = None;
    for p_path in libraries {
        // Find a filename that matches the pattern: (lib_prefix)-hash-(version)(lib_suffix)
        // and remember what the hash was
        let mut f_name = match p_path.filestem_str() {
            Some(s) => s, None => continue
        };
        // Already checked the filetype above

         // This is complicated because library names and versions can both contain dashes
         loop {
            if f_name.is_empty() { break; }
            match f_name.rfind('-') {
                Some(i) => {
                    debug!("Maybe {} is a version", f_name.slice(i + 1, f_name.len()));
                    match try_parsing_version(f_name.slice(i + 1, f_name.len())) {
                       Some(ref found_vers) if version == found_vers => {
                           match f_name.slice(0, i).rfind('-') {
                               Some(j) => {
                                   debug!("Maybe {} equals {}", f_name.slice(0, j), lib_prefix);
                                   if f_name.slice(0, j) == lib_prefix {
                                       result_filename = Some(p_path.clone());
                                   }
                                   break;
                               }
                               None => break
                           }

                       }
                       _ => { f_name = f_name.slice(0, i); }
                 }
               }
               None => break
         } // match
       } // loop
    } // for

    if result_filename.is_none() {
        debug!("warning: library_in_workspace didn't find a library in {} for {}",
                  dir_to_search.display(), short_name);
    }

    // Return the filename that matches, which we now know exists
    // (if result_filename != None)
    let abs_path = result_filename.map(|result_filename| {
        let absolute_path = dir_to_search.join(&result_filename);
        debug!("result_filename = {}", absolute_path.display());
        absolute_path
    });

    abs_path
}

fn split_crate_id<'a>(crate_id: &'a str) -> (&'a str, Version) {
    match split_version(crate_id) {
        Some((name, vers)) =>
            match vers {
                ExactRevision(ref v) => match v.find('-') {
                    Some(pos) => (name, ExactRevision(v.slice(0, pos).to_owned())),
                    None => (name, ExactRevision(v.to_owned()))
                },
                _ => (name, vers)
            },
        None => (crate_id, NoVersion)
    }
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
                (Build, _)      => target_build_dir(workspace).join(&crateid.path),
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
    result.push(&crateid.path);
    debug!("Creating build dir {} for package id {}", result.display(),
           crateid.to_str());
    fs::mkdir_recursive(&result, io::UserRWX);
    return result;
}

/// Return the output file for a given directory name,
/// given whether we're building a library and whether we're building tests
pub fn mk_output_path(what: OutputType, where: Target,
                      pkg_id: &CrateId, workspace: Path) -> Path {
    let short_name_with_version = format!("{}-{}", pkg_id.short_name,
                                          pkg_id.version.to_str());
    // Not local_path.dir_path()! For package foo/bar/blat/, we want
    // the executable blat-0.5 to live under blat/
    let dir = match where {
        // If we're installing, it just goes under <workspace>...
        Install => workspace,
        // and if we're just building, it goes in a package-specific subdir
        Build => workspace.join(&pkg_id.path)
    };
    debug!("[{:?}:{:?}] mk_output_path: short_name = {}, path = {}", what, where,
           if what == Lib { short_name_with_version.clone() } else { pkg_id.short_name.clone() },
           dir.display());
    let mut output_path = match what {
        // this code is duplicated from elsewhere; fix this
        Lib => dir.join(os::dll_filename(short_name_with_version)),
        // executable names *aren't* versioned
        _ => dir.join(format!("{}{}{}", pkg_id.short_name,
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
    for dir in rp.iter() {
        // Require that the parent directory match the package ID
        // Note that this only matches if the package ID being searched for
        // has a name that's a single component
        if dir.ends_with_path(&p.path) || dir.ends_with_path(&versionize(&p.path, &p.version)) {
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
pub fn versionize(p: &Path, v: &Version) -> Path {
    let q = p.filename().expect("path is a directory");
    let mut q = q.to_owned();
    q.push('-' as u8);
    let vs = v.to_str();
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
