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

use std::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
use std::os::mkdir_recursive;
use std::os;
use messages::*;

pub fn default_workspace() -> Path {
    let p = rust_path();
    if p.is_empty() {
        fail!("Empty RUST_PATH");
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
    workspace_contains_package_id_(pkgid, workspace, |p| { p.push("src") }).is_some()
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
            if *p == src_dir.push_rel(&pkgid.path) || {
                let pf = p.filename();
                do pf.iter().any |pf| {
                    let g = pf.to_str();
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
        debug!("Found %s in %s", pkgid.to_str(), workspace.to_str());
    } else {
        debug!("Didn't find %s in %s", pkgid.to_str(), workspace.to_str());
    }
    found
}

/// Return the target-specific build subdirectory, pushed onto `base`;
/// doesn't check that it exists or create it
pub fn target_build_dir(workspace: &Path) -> Path {
    workspace.push("build").push(host_triple())
}

/// Return the target-specific lib subdirectory, pushed onto `base`;
/// doesn't check that it exists or create it
fn target_lib_dir(workspace: &Path) -> Path {
    workspace.push("lib").push(host_triple())
}

/// Return the bin subdirectory, pushed onto `base`;
/// doesn't check that it exists or create it
/// note: this isn't target-specific
fn target_bin_dir(workspace: &Path) -> Path {
    workspace.push("bin")
}

/// Figure out what the executable name for <pkgid> in <workspace>'s build
/// directory is, and if the file exists, return it.
pub fn built_executable_in_workspace(pkgid: &PkgId, workspace: &Path) -> Option<Path> {
    let mut result = target_build_dir(workspace);
    result = mk_output_path(Main, Build, pkgid, result);
    debug!("built_executable_in_workspace: checking whether %s exists",
           result.to_str());
    if os::path_exists(&result) {
        Some(result)
    }
    else {
        debug!("built_executable_in_workspace: %s does not exist", result.to_str());
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
    library_in_workspace(&pkgid.path, pkgid.short_name, Build, workspace, "build", &pkgid.version)
}

/// Does the actual searching stuff
pub fn installed_library_in_workspace(pkg_path: &Path, workspace: &Path) -> Option<Path> {
    // This could break once we're handling multiple versions better -- I should add a test for it
    match pkg_path.filename() {
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
    debug!("library_in_workspace: checking whether a library named %s exists",
           short_name);

    // We don't know what the hash is, so we have to search through the directory
    // contents

    debug!("short_name = %s where = %? workspace = %s \
            prefix = %s", short_name, where, workspace.to_str(), prefix);

    let dir_to_search = match where {
        Build => target_build_dir(workspace).push_rel(path),
        Install => target_lib_dir(workspace)
    };

    library_in(short_name, version, &dir_to_search)
}

// rustc doesn't use target-specific subdirectories
pub fn system_library(sysroot: &Path, lib_name: &str) -> Option<Path> {
    library_in(lib_name, &NoVersion, &sysroot.push("lib"))
}

fn library_in(short_name: &str, version: &Version, dir_to_search: &Path) -> Option<Path> {
    debug!("Listing directory %s", dir_to_search.to_str());
    let dir_contents = os::list_dir(dir_to_search);
    debug!("dir has %? entries", dir_contents.len());

    let lib_prefix = fmt!("%s%s", os::consts::DLL_PREFIX, short_name);
    let lib_filetype = os::consts::DLL_SUFFIX;

    debug!("lib_prefix = %s and lib_filetype = %s", lib_prefix, lib_filetype);

    // Find a filename that matches the pattern:
    // (lib_prefix)-hash-(version)(lib_suffix)
    let paths = do dir_contents.iter().map |p| {
        Path((*p).clone())
    };

    let mut libraries = do paths.filter |p| {
        let extension = p.filetype();
        debug!("p = %s, p's extension is %?", p.to_str(), extension);
        match extension {
            None => false,
            Some(ref s) => lib_filetype == *s
        }
    };

    let mut result_filename = None;
    for p_path in libraries {
        // Find a filename that matches the pattern: (lib_prefix)-hash-(version)(lib_suffix)
        // and remember what the hash was
        let mut f_name = match p_path.filestem() {
            Some(s) => s, None => loop
        };
        // Already checked the filetype above

         // This is complicated because library names and versions can both contain dashes
         loop {
            if f_name.is_empty() { break; }
            match f_name.rfind('-') {
                Some(i) => {
                    debug!("Maybe %s is a version", f_name.slice(i + 1, f_name.len()));
                    match try_parsing_version(f_name.slice(i + 1, f_name.len())) {
                       Some(ref found_vers) if version == found_vers => {
                           match f_name.slice(0, i).rfind('-') {
                               Some(j) => {
                                   debug!("Maybe %s equals %s", f_name.slice(0, j), lib_prefix);
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
        debug!("warning: library_in_workspace didn't find a library in %s for %s",
                  dir_to_search.to_str(), short_name);
    }

    // Return the filename that matches, which we now know exists
    // (if result_filename != None)
    let abs_path = do result_filename.map |result_filename| {
        let absolute_path = dir_to_search.push_rel(result_filename);
        debug!("result_filename = %s", absolute_path.to_str());
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
                    fmt!("Workspace supplied to target_library_in_workspace \
                          is not a directory! %s", workspace.to_str())));
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
                (Build, _)         => target_build_dir(workspace).push_rel(&pkgid.path),
                (Install, Lib)     => target_lib_dir(workspace),
                (Install, _)    => target_bin_dir(workspace)
    };
    if !os::path_exists(&result) && !mkdir_recursive(&result, U_RWX) {
        cond.raise((result.clone(), fmt!("target_file_in_workspace couldn't \
            create the %s dir (pkgid=%s, workspace=%s, what=%?, where=%?",
            subdir, pkgid.to_str(), workspace.to_str(), what, where)));
    }
    mk_output_path(what, where, pkgid, result)
}

/// Return the directory for <pkgid>'s build artifacts in <workspace>.
/// Creates it if it doesn't exist.
pub fn build_pkg_id_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    use conditions::bad_path::cond;

    let mut result = target_build_dir(workspace);
    result = result.push_rel(&pkgid.path);
    debug!("Creating build dir %s for package id %s", result.to_str(),
           pkgid.to_str());
    if os::path_exists(&result) || os::mkdir_recursive(&result, U_RWX) {
        result
    }
    else {
        cond.raise((result, fmt!("Could not create directory for package %s", pkgid.to_str())))
    }
}

/// Return the output file for a given directory name,
/// given whether we're building a library and whether we're building tests
pub fn mk_output_path(what: OutputType, where: Target,
                      pkg_id: &PkgId, workspace: Path) -> Path {
    let short_name_with_version = fmt!("%s-%s", pkg_id.short_name,
                                       pkg_id.version.to_str());
    // Not local_path.dir_path()! For package foo/bar/blat/, we want
    // the executable blat-0.5 to live under blat/
    let dir = match where {
        // If we're installing, it just goes under <workspace>...
        Install => workspace,
        // and if we're just building, it goes in a package-specific subdir
        Build => workspace.push_rel(&pkg_id.path)
    };
    debug!("[%?:%?] mk_output_path: short_name = %s, path = %s", what, where,
           if what == Lib { short_name_with_version.clone() } else { pkg_id.short_name.clone() },
           dir.to_str());
    let mut output_path = match what {
        // this code is duplicated from elsewhere; fix this
        Lib => dir.push(os::dll_filename(short_name_with_version)),
        // executable names *aren't* versioned
        _ => dir.push(fmt!("%s%s%s", pkg_id.short_name,
                           match what {
                               Test => "test",
                               Bench => "bench",
                               _     => ""
                           },
                           os::EXE_SUFFIX))
    };
    if !output_path.is_absolute() {
        output_path = os::getcwd().push_rel(&output_path).normalize();
    }
    debug!("mk_output_path: returning %s", output_path.to_str());
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
        warn(fmt!("Warning: there don't seem to be any files for %s installed in %s",
             pkgid.to_str(), workspace.to_str()));
    }

}

fn dir_has_file(dir: &Path, file: &str) -> bool {
    assert!(dir.is_absolute());
    os::path_exists(&dir.push(file))
}

pub fn find_dir_using_rust_path_hack(p: &PkgId) -> Option<Path> {
    let rp = rust_path();
    for dir in rp.iter() {
        debug!("In find_dir_using_rust_path_hack: checking dir %s", dir.to_str());
        if dir_has_file(dir, "lib.rs") || dir_has_file(dir, "main.rs")
            || dir_has_file(dir, "test.rs") || dir_has_file(dir, "bench.rs") {
            debug!("Did find id %s in dir %s", p.to_str(), dir.to_str());
            return Some(dir.clone());
        }
        debug!("Didn't find id %s in dir %s", p.to_str(), dir.to_str())
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
