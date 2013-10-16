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

pub use package_id::PkgId;
pub use target::{OutputType, Main, Lib, Test, Bench, Target, Build, Install};
pub use version::{Version, NoVersion, split_version_general, try_parsing_version};
pub use rustc::metadata::filesearch::rust_path;
use rustc::driver::driver::host_triple;

use std::libc;
use std::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
use std::os::mkdir_recursive;
use std::os;
use messages::*;

pub fn default_workspace() -> Path {
    let p = rust_path();
    if p.is_empty() {
        fail2!("Empty RUST_PATH");
    }
    let result = p[0];
    if !os::path_is_dir(&result) {
        os::mkdir_recursive(&result, U_RWX);
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
pub fn make_dir_rwx(p: &Path) -> bool { os::make_dir(p, U_RWX) }

pub fn make_dir_rwx_recursive(p: &Path) -> bool { os::mkdir_recursive(p, U_RWX) }

// n.b. The next three functions ignore the package version right
// now. Should fix that.

/// True if there's a directory in <workspace> with
/// pkgid's short name
pub fn workspace_contains_package_id(pkgid: &PkgId, workspace: &Path) -> bool {
    workspace_contains_package_id_(pkgid, workspace, |p| p.join("src")).is_some()
}

pub fn workspace_contains_package_id_(pkgid: &PkgId, workspace: &Path,
// Returns the directory it was actually found in
             workspace_to_src_dir: &fn(&Path) -> Path) -> Option<Path> {
    if !os::path_is_dir(workspace) {
        return None;
    }

    let src_dir = workspace_to_src_dir(workspace);

    let mut found = None;
    do os::walk_dir(&src_dir) |p| {
        if os::path_is_dir(p) {
            if *p == src_dir.join(&pkgid.path) || {
                let pf = p.filename_str();
                do pf.iter().any |&g| {
                    match split_version_general(g, '-') {
                        None => false,
                        Some((ref might_match, ref vers)) => {
                            *might_match == pkgid.short_name
                                && (pkgid.version == *vers || pkgid.version == NoVersion)
                        }
                    }
                }
            } {
                found = Some(p.clone());
            }

        };
        true
    };

    if found.is_some() {
        debug2!("Found {} in {}", pkgid.to_str(), workspace.display());
    } else {
        debug2!("Didn't find {} in {}", pkgid.to_str(), workspace.display());
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
    let mut dir = workspace.join("lib");
    dir.push(host_triple());
    dir
}

/// Return the bin subdirectory, pushed onto `base`;
/// doesn't check that it exists or create it
/// note: this isn't target-specific
fn target_bin_dir(workspace: &Path) -> Path {
    workspace.join("bin")
}

/// Figure out what the executable name for <pkgid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_executable_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
    let mut result = target_build_dir(workspace);
    result = mk_output_path(Main, Build, pkgid, result);
    debug2!("built_executable_in_workspace: checking whether {} exists",
           result.display());
    if os::path_exists(&result) {
        Some(result)
    }
    else {
        debug2!("built_executable_in_workspace: {} does not exist", result.display());
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
    let mut result = target_build_dir(workspace);
    // should use a target-specific subdirectory
    result = mk_output_path(what, Build, pkgid, result);
    debug2!("output_in_workspace: checking whether {} exists",
           result.display());
    if os::path_exists(&result) {
        Some(result)
    }
    else {
        error2!("output_in_workspace: {} does not exist", result.display());
        None
    }
}

/// Figure out what the library name for <pkgid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_library_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
    library_in_workspace(&pkgid.path, pkgid.short_name, Build, workspace, "build", &pkgid.version)
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
                                                 "lib",
                                                 &NoVersion)
    }
}

/// `workspace` is used to figure out the directory to search.
/// `short_name` is taken as the link name of the library.
pub fn library_in_workspace(path: &Path, short_name: &str, where: Target,
                        workspace: &Path, prefix: &str, version: &Version) -> Option<Path> {
    debug2!("library_in_workspace: checking whether a library named {} exists",
           short_name);

    // We don't know what the hash is, so we have to search through the directory
    // contents

    debug2!("short_name = {} where = {:?} workspace = {} \
            prefix = {}", short_name, where, workspace.display(), prefix);

    let dir_to_search = match where {
        Build => target_build_dir(workspace).join(path),
        Install => target_lib_dir(workspace)
    };

    library_in(short_name, version, &dir_to_search)
}

// rustc doesn't use target-specific subdirectories
pub fn system_library(sysroot: &Path, lib_name: &str) -> Option<Path> {
    library_in(lib_name, &NoVersion, &sysroot.join("lib"))
}

fn library_in(short_name: &str, version: &Version, dir_to_search: &Path) -> Option<Path> {
    debug2!("Listing directory {}", dir_to_search.display());
    let dir_contents = os::list_dir(dir_to_search);
    debug2!("dir has {:?} entries", dir_contents.len());

    let lib_prefix = format!("{}{}", os::consts::DLL_PREFIX, short_name);
    let lib_filetype = os::consts::DLL_EXTENSION;

    debug2!("lib_prefix = {} and lib_filetype = {}", lib_prefix, lib_filetype);

    // Find a filename that matches the pattern:
    // (lib_prefix)-hash-(version)(lib_suffix)
    let mut libraries = do dir_contents.iter().filter |p| {
        let extension = p.extension_str();
        debug2!("p = {}, p's extension is {:?}", p.display(), extension);
        match extension {
            None => false,
            Some(ref s) => lib_filetype == *s
        }
    };

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
                    debug2!("Maybe {} is a version", f_name.slice(i + 1, f_name.len()));
                    match try_parsing_version(f_name.slice(i + 1, f_name.len())) {
                       Some(ref found_vers) if version == found_vers => {
                           match f_name.slice(0, i).rfind('-') {
                               Some(j) => {
                                   debug2!("Maybe {} equals {}", f_name.slice(0, j), lib_prefix);
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
        debug2!("warning: library_in_workspace didn't find a library in {} for {}",
                  dir_to_search.display(), short_name);
    }

    // Return the filename that matches, which we now know exists
    // (if result_filename != None)
    let abs_path = do result_filename.map |result_filename| {
        let absolute_path = dir_to_search.join(&result_filename);
        debug2!("result_filename = {}", absolute_path.display());
        absolute_path
    };

    abs_path
}

/// Returns the executable that would be installed for <pkgid>
/// in <workspace>
/// As a side effect, creates the bin-dir if it doesn't exist
pub fn target_executable_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    target_file_in_workspace(pkgid, workspace, Main, Install)
}


/// Returns the executable that would be installed for <pkgid>
/// in <workspace>
/// As a side effect, creates the lib-dir if it doesn't exist
pub fn target_library_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    use conditions::bad_path::cond;
    if !os::path_is_dir(workspace) {
        cond.raise(((*workspace).clone(),
                    format!("Workspace supplied to target_library_in_workspace \
                             is not a directory! {}", workspace.display())));
    }
    target_file_in_workspace(pkgid, workspace, Lib, Install)
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
    // Artifacts in the build directory live in a package-ID-specific subdirectory,
    // but installed ones don't.
    let result = match (where, what) {
                (Build, _)      => target_build_dir(workspace).join(&pkgid.path),
                (Install, Lib)  => target_lib_dir(workspace),
                (Install, _)    => target_bin_dir(workspace)
    };
    if !os::path_exists(&result) && !mkdir_recursive(&result, U_RWX) {
        cond.raise((result.clone(), format!("target_file_in_workspace couldn't \
            create the {} dir (pkgid={}, workspace={}, what={:?}, where={:?}",
            subdir, pkgid.to_str(), workspace.display(), what, where)));
    }
    mk_output_path(what, where, pkgid, result)
}

/// Return the directory for <pkgid>'s build artifacts in <workspace>.
/// Creates it if it doesn't exist.
pub fn build_pkg_id_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    use conditions::bad_path::cond;

    let mut result = target_build_dir(workspace);
    result.push(&pkgid.path);
    debug2!("Creating build dir {} for package id {}", result.display(),
           pkgid.to_str());
    if os::path_exists(&result) || os::mkdir_recursive(&result, U_RWX) {
        result
    }
    else {
        cond.raise((result, format!("Could not create directory for package {}", pkgid.to_str())))
    }
}

/// Return the output file for a given directory name,
/// given whether we're building a library and whether we're building tests
pub fn mk_output_path(what: OutputType, where: Target,
                      pkg_id: &PkgId, workspace: Path) -> Path {
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
    debug2!("[{:?}:{:?}] mk_output_path: short_name = {}, path = {}", what, where,
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
                           os::EXE_SUFFIX))
    };
    if !output_path.is_absolute() {
        output_path = os::getcwd().join(&output_path);
    }
    debug2!("mk_output_path: returning {}", output_path.display());
    output_path
}

/// Removes files for the package `pkgid`, assuming it's installed in workspace `workspace`
pub fn uninstall_package_from(workspace: &Path, pkgid: &PkgId) {
    let mut did_something = false;
    let installed_bin = target_executable_in_workspace(pkgid, workspace);
    if os::path_exists(&installed_bin) {
        os::remove_file(&installed_bin);
        did_something = true;
    }
    let installed_lib = target_library_in_workspace(pkgid, workspace);
    if os::path_exists(&installed_lib) {
        os::remove_file(&installed_lib);
        did_something = true;
    }
    if !did_something {
        warn(format!("Warning: there don't seem to be any files for {} installed in {}",
             pkgid.to_str(), workspace.display()));
    }

}

fn dir_has_file(dir: &Path, file: &str) -> bool {
    assert!(dir.is_absolute());
    os::path_exists(&dir.join(file))
}

pub fn find_dir_using_rust_path_hack(p: &PkgId) -> Option<Path> {
    let rp = rust_path();
    for dir in rp.iter() {
        // Require that the parent directory match the package ID
        // Note that this only matches if the package ID being searched for
        // has a name that's a single component
        if dir.ends_with_path(&p.path) || dir.ends_with_path(&versionize(&p.path, &p.version)) {
            debug2!("In find_dir_using_rust_path_hack: checking dir {}", dir.display());
            if dir_has_file(dir, "lib.rs") || dir_has_file(dir, "main.rs")
                || dir_has_file(dir, "test.rs") || dir_has_file(dir, "bench.rs") {
                debug2!("Did find id {} in dir {}", p.to_str(), dir.display());
                return Some(dir.clone());
            }
        }
        debug2!("Didn't find id {} in dir {}", p.to_str(), dir.display())
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
    #[fixed_stack_segment];
    unsafe {
        do p.with_c_str |src_buf| {
            libc::chmod(src_buf, S_IRUSR as libc::c_int) == 0 as libc::c_int
        }
    }
}

#[cfg(not(target_os = "win32"))]
pub fn chmod_read_only(p: &Path) -> bool {
    #[fixed_stack_segment];
    unsafe {
        do p.with_c_str |src_buf| {
            libc::chmod(src_buf, S_IRUSR as libc::mode_t) == 0
                as libc::c_int
        }
    }
}

