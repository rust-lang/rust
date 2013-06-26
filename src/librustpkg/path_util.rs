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
pub use package_path::{RemotePath, LocalPath, normalize};
pub use package_id::PkgId;
pub use target::{OutputType, Main, Lib, Test, Bench, Target, Build, Install};
pub use version::{Version, NoVersion, split_version_general};
use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};
use core::os::mkdir_recursive;
use core::os;
use core::iterator::IteratorUtil;
use messages::*;
use package_id::*;

fn push_if_exists(vec: &mut ~[Path], p: &Path) {
    let maybe_dir = p.push(".rust");
    if os::path_exists(&maybe_dir) {
        vec.push(maybe_dir);
    }
}

#[cfg(windows)]
static path_entry_separator: &'static str = ";";
#[cfg(not(windows))]
static path_entry_separator: &'static str = ":";

/// Returns the value of RUST_PATH, as a list
/// of Paths. Includes default entries for, if they exist:
/// $HOME/.rust
/// DIR/.rust for any DIR that's the current working directory
/// or an ancestor of it
pub fn rust_path() -> ~[Path] {
    let env_path: ~str = os::getenv("RUST_PATH").get_or_default(~"");
    let mut env_rust_path: ~[Path] = match os::getenv("RUST_PATH") {
        Some(env_path) => {
            let env_path_components: ~[&str] =
                env_path.split_str_iter(path_entry_separator).collect();
            env_path_components.map(|&s| Path(s))
        }
        None => ~[]
    };
    let cwd = os::getcwd();
    // now add in default entries
    env_rust_path.push(copy cwd);
    do cwd.each_parent() |p| { push_if_exists(&mut env_rust_path, p) };
    let h = os::homedir();
    for h.iter().advance |h| { push_if_exists(&mut env_rust_path, h); }
    env_rust_path
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
    let src_dir = workspace.push("src");
    let dirs = os::list_dir(&src_dir);
    for dirs.iter().advance |&p| {
        let p = Path(p);
        debug!("=> p = %s", p.to_str());
        if !os::path_is_dir(&src_dir.push_rel(&p)) {
            loop;
        }
        debug!("p = %s, remote_path = %s", p.to_str(), pkgid.remote_path.to_str());

        if p == *pkgid.remote_path {
            return true;
        }
        else {
            let pf = p.filename();
            for pf.iter().advance |&pf| {
                let f_ = copy pf;
                let g = f_.to_str();
                match split_version_general(g, '-') {
                    Some((ref might_match, ref vers)) => {
                        debug!("might_match = %s, vers = %s", *might_match,
                               vers.to_str());
                        if *might_match == pkgid.short_name
                            && (*vers == pkgid.version || pkgid.version == NoVersion)
                        {
                            return true;
                        }
                    }
                    None => ()
                }
            }
        }
    }
    false
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
    for rs.iter().advance |p| {
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
    library_in_workspace(&pkgid.local_path, pkgid.short_name,
                         Build, workspace, "build")
}

/// Does the actual searching stuff
pub fn installed_library_in_workspace(short_name: &str, workspace: &Path) -> Option<Path> {
    library_in_workspace(&normalize(RemotePath(Path(short_name))),
                         short_name, Install, workspace, "lib")
}


/// This doesn't take a PkgId, so we can use it for `extern mod` inference, where we
/// don't know the entire package ID.
/// `workspace` is used to figure out the directory to search.
/// `short_name` is taken as the link name of the library.
pub fn library_in_workspace(path: &LocalPath, short_name: &str, where: Target,
                        workspace: &Path, prefix: &str) -> Option<Path> {
    debug!("library_in_workspace: checking whether a library named %s exists",
           short_name);

    // We don't know what the hash is, so we have to search through the directory
    // contents

    debug!("short_name = %s where = %? workspace = %s \
            prefix = %s", short_name, where, workspace.to_str(), prefix);

    let dir_to_search = match where {
        Build => workspace.push(prefix).push_rel(&**path),
        Install => workspace.push(prefix)
    };
    debug!("Listing directory %s", dir_to_search.to_str());
    let dir_contents = os::list_dir(&dir_to_search);
    debug!("dir has %? entries", dir_contents.len());

    let lib_prefix = fmt!("%s%s", os::consts::DLL_PREFIX, short_name);
    let lib_filetype = os::consts::DLL_SUFFIX;

    debug!("lib_prefix = %s and lib_filetype = %s", lib_prefix, lib_filetype);

    let mut result_filename = None;
    for dir_contents.iter().advance |&p| {
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
        None => {
            warn(fmt!("library_in_workspace didn't find a library in %s for %s",
                            dir_to_search.to_str(), short_name));
            None
        }
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


/// Returns the executable that would be installed for <pkgid>
/// in <workspace>
/// As a side effect, creates the lib-dir if it doesn't exist
pub fn target_library_in_workspace(pkgid: &PkgId, workspace: &Path) -> Path {
    use conditions::bad_path::cond;
    if !os::path_is_dir(workspace) {
        cond.raise((copy *workspace,
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
    let result = workspace.push(subdir);
    if !os::path_exists(&result) && !mkdir_recursive(&result, u_rwx) {
        cond.raise((copy result, fmt!("target_file_in_workspace couldn't \
            create the %s dir (pkgid=%s, workspace=%s, what=%?, where=%?",
            subdir, pkgid.to_str(), workspace.to_str(), what, where)));
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
    let short_name_with_version = fmt!("%s-%s", pkg_id.short_name,
                                       pkg_id.version.to_str());
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
        _ => dir.push(fmt!("%s%s%s", pkg_id.short_name,
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
