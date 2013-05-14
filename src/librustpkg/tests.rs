// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustpkg unit tests

use context::Ctx;
use core::hashmap::HashMap;
use core::path::Path;
use std::tempfile::mkdtemp;
use util::{PkgId, default_version};
use path_util::{target_executable_in_workspace, target_library_in_workspace,
               target_test_in_workspace, target_bench_in_workspace,
               make_dir_rwx, u_rwx};
use core::os::mkdir_recursive;

fn fake_ctxt(sysroot_opt: Option<@Path>) -> Ctx {
    Ctx {
        sysroot_opt: sysroot_opt,
        json: false,
        dep_cache: @mut HashMap::new()
    }
}

fn fake_pkg() -> PkgId {
    PkgId {
        path: Path(~"bogus"),
        version: default_version()
    }
}

fn remote_pkg() -> PkgId {
    PkgId {
        path: Path(~"github.com/catamorphism/test-pkg"),
        version: default_version()
    }
}

fn writeFile(file_path: &Path, contents: ~str) {
    let out: @io::Writer =
        result::get(&io::file_writer(file_path,
                                     ~[io::Create, io::Truncate]));
    out.write_line(contents);
}

fn mk_temp_workspace(short_name: &Path) -> Path {
    let workspace = mkdtemp(&os::tmpdir(), "test").expect("couldn't create temp dir");
    let package_dir = workspace.push(~"src").push_rel(short_name);
    assert!(mkdir_recursive(&package_dir, u_rwx));
    // Create main, lib, test, and bench files
    writeFile(&package_dir.push(~"main.rs"),
              ~"fn main() { let _x = (); }");
    writeFile(&package_dir.push(~"lib.rs"),
              ~"pub fn f() { let _x = (); }");
    writeFile(&package_dir.push(~"test.rs"),
              ~"#[test] pub fn f() { (); }");
    writeFile(&package_dir.push(~"bench.rs"),
              ~"#[bench] pub fn f() { (); }");
    workspace
}

fn is_rwx(p: &Path) -> bool {
    use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};

    match p.get_mode() {
        None => return false,
        Some(m) =>
            ((m & S_IRUSR as uint) == S_IRUSR as uint
            && (m & S_IWUSR as uint) == S_IWUSR as uint
            && (m & S_IXUSR as uint) == S_IXUSR as uint)
    }
}

fn test_sysroot() -> Path {
    // Totally gross hack but it's just for test cases.
    // Infer the sysroot from the exe name and tack "stage2"
    // onto it. (Did I mention it was a gross hack?)
    let self_path = os::self_exe_path().expect("Couldn't get self_exe path");
    self_path.pop().push("stage2")
}

#[test]
fn test_make_dir_rwx() {
    let temp = &os::tmpdir();
    let dir = temp.push(~"quux");
    assert!(!os::path_exists(&dir) ||
            os::remove_dir_recursive(&dir));
    debug!("Trying to make %s", dir.to_str());
    assert!(make_dir_rwx(&dir));
    assert!(os::path_is_dir(&dir));
    assert!(is_rwx(&dir));
    assert!(os::remove_dir_recursive(&dir));
}

#[test]
fn test_install_valid() {
    let sysroot = test_sysroot();
    debug!("sysroot = %s", sysroot.to_str());
    let ctxt = fake_ctxt(Some(@sysroot));
    let temp_pkg_id = fake_pkg();
    let temp_workspace = mk_temp_workspace(&temp_pkg_id.path);
    // should have test, bench, lib, and main
    ctxt.install(&temp_workspace, &temp_pkg_id);
    // Check that all files exist
    let exec = target_executable_in_workspace(&temp_pkg_id, &temp_workspace);
    debug!("exec = %s", exec.to_str());
    assert!(os::path_exists(&exec));
    assert!(is_rwx(&exec));
    let lib = target_library_in_workspace(&temp_pkg_id, &temp_workspace);
    debug!("lib = %s", lib.to_str());
    assert!(os::path_exists(&lib));
    assert!(is_rwx(&lib));
    // And that the test and bench executables aren't installed
    assert!(!os::path_exists(&target_test_in_workspace(&temp_pkg_id, &temp_workspace)));
    let bench = target_bench_in_workspace(&temp_pkg_id, &temp_workspace);
    debug!("bench = %s", bench.to_str());
    assert!(!os::path_exists(&bench));
}

#[test]
fn test_install_invalid() {
    use conditions::nonexistent_package::cond;
    use cond1 = conditions::missing_pkg_files::cond;

    let ctxt = fake_ctxt(None);
    let pkgid = fake_pkg();
    let temp_workspace = mkdtemp(&os::tmpdir(), "test").expect("couldn't create temp dir");
    let mut error_occurred = false;
    let mut error1_occurred = false;
    do cond1.trap(|_| {
        error1_occurred = true;
    }).in {
        do cond.trap(|_| {
            error_occurred = true;
        }).in {
            ctxt.install(&temp_workspace, &pkgid);
        }
    }
    assert!(error_occurred && error1_occurred);
}

#[test]
#[ignore(reason = "install from URL-fragment not yet implemented")]
fn test_install_url() {
    let sysroot = test_sysroot();
    debug!("sysroot = %s", sysroot.to_str());
    let ctxt = fake_ctxt(Some(@sysroot));
    let temp_pkg_id = remote_pkg();
    let temp_workspace = mk_temp_workspace(&temp_pkg_id.path);
    // should have test, bench, lib, and main
    ctxt.install(&temp_workspace, &temp_pkg_id);
    // Check that all files exist
    let exec = target_executable_in_workspace(&temp_pkg_id, &temp_workspace);
    debug!("exec = %s", exec.to_str());
    assert!(os::path_exists(&exec));
    assert!(is_rwx(&exec));
    let lib = target_library_in_workspace(&temp_pkg_id, &temp_workspace);
    debug!("lib = %s", lib.to_str());
    assert!(os::path_exists(&lib));
    assert!(is_rwx(&lib));
    // And that the test and bench executables aren't installed
    assert!(!os::path_exists(&target_test_in_workspace(&temp_pkg_id, &temp_workspace)));
    let bench = target_bench_in_workspace(&temp_pkg_id, &temp_workspace);
    debug!("bench = %s", bench.to_str());
    assert!(!os::path_exists(&bench));
}
