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
               make_dir_rwx};

fn fake_ctxt() -> Ctx {
    Ctx {
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

fn mk_temp_workspace() -> Path {
    mkdtemp(&os::tmpdir(), "test").expect("couldn't create temp dir")
}

fn is_rwx(p: &Path) -> bool {
    use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};

    match p.get_mode() {
        None => return false,
        Some(m) => {
            ((m & S_IRUSR as uint) == S_IRUSR as uint
            && (m & S_IWUSR as uint) == S_IWUSR as uint
            && (m & S_IXUSR as uint) == S_IXUSR as uint)
        }
    }
}

#[test]
fn test_make_dir_rwx() {
    let temp = &os::tmpdir();
    let dir = temp.push(~"quux");
    let _ = os::remove_dir(&dir);
    assert!(make_dir_rwx(&dir));
    assert!(os::path_is_dir(&dir));
    assert!(is_rwx(&dir));
    assert!(os::remove_dir(&dir));
}

#[test]
#[ignore(reason = "install not yet implemented")]
fn test_install_valid() {
    let ctxt = fake_ctxt();
    let temp_pkg_id = fake_pkg();
    let temp_workspace = mk_temp_workspace();
    // should have test, bench, lib, and main
    ctxt.install(&temp_workspace, temp_pkg_id);
    // Check that all files exist
    let exec = target_executable_in_workspace(temp_pkg_id, &temp_workspace);
    assert!(os::path_exists(&exec));
    assert!(is_rwx(&exec));
    let lib = target_library_in_workspace(temp_pkg_id, &temp_workspace);
    assert!(os::path_exists(&lib));
    assert!(is_rwx(&lib));
    // And that the test and bench executables aren't installed
    assert!(!os::path_exists(&target_test_in_workspace(temp_pkg_id, &temp_workspace)));
    assert!(!os::path_exists(&target_bench_in_workspace(temp_pkg_id, &temp_workspace)));
}

#[test]
#[ignore(reason = "install not yet implemented")]
fn test_install_invalid() {
    use conditions::nonexistent_package::cond;

    let ctxt = fake_ctxt();
    let pkgid = fake_pkg();
    let temp_workspace = mk_temp_workspace();
    let expected_path = Path(~"quux");
    let substituted: Path = do cond.trap(|_| {
        expected_path
    }).in {
        ctxt.install(&temp_workspace, pkgid);
        // ok
        fail!(~"test_install_invalid failed, should have raised a condition");
    };
    assert!(substituted == expected_path);
}
