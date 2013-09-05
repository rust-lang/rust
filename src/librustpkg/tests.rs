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
use std::hashmap::HashMap;
use std::{io, libc, os, run, str};
use extra::tempfile::mkdtemp;
use std::run::ProcessOutput;
use installed_packages::list_installed_packages;
use package_id::{PkgId};
use version::{ExactRevision, NoVersion, Version, Tagged};
use path_util::{target_executable_in_workspace, target_test_in_workspace,
               target_bench_in_workspace, make_dir_rwx, U_RWX,
               library_in_workspace, installed_library_in_workspace,
               built_bench_in_workspace, built_test_in_workspace,
               built_library_in_workspace, built_executable_in_workspace};
use rustc::metadata::filesearch::rust_path;
use rustc::driver::driver::host_triple;
use target::*;

/// Returns the last-modified date as an Option
fn datestamp(p: &Path) -> Option<libc::time_t> {
    p.stat().map(|stat| stat.st_mtime)
}

fn fake_ctxt(sysroot_opt: Option<@Path>) -> Ctx {
    Ctx {
        use_rust_path_hack: false,
        sysroot_opt: sysroot_opt,
        json: false,
        dep_cache: @mut HashMap::new()
    }
}

fn fake_pkg() -> PkgId {
    let sn = ~"bogus";
    PkgId {
        path: Path(sn),
        short_name: sn,
        version: NoVersion
    }
}

fn git_repo_pkg() -> PkgId {
    PkgId {
        path: Path("mockgithub.com/catamorphism/test-pkg"),
        short_name: ~"test-pkg",
        version: NoVersion
    }
}

fn git_repo_pkg_with_tag(a_tag: ~str) -> PkgId {
    PkgId {
        path: Path("mockgithub.com/catamorphism/test-pkg"),
        short_name: ~"test-pkg",
        version: Tagged(a_tag)
    }
}

fn writeFile(file_path: &Path, contents: &str) {
    let out = io::file_writer(file_path, [io::Create, io::Truncate]).unwrap();
    out.write_line(contents);
}

fn mk_empty_workspace(short_name: &Path, version: &Version, tag: &str) -> Path {
    let workspace_dir = mkdtemp(&os::tmpdir(), tag).expect("couldn't create temp dir");
    mk_workspace(&workspace_dir, short_name, version);
    workspace_dir
}

fn mk_workspace(workspace: &Path, short_name: &Path, version: &Version) -> Path {
    // include version number in directory name
    let package_dir = workspace.push("src").push(fmt!("%s-%s",
                                                      short_name.to_str(), version.to_str()));
    assert!(os::mkdir_recursive(&package_dir, U_RWX));
    package_dir
}

fn mk_temp_workspace(short_name: &Path, version: &Version) -> Path {
    let package_dir = mk_empty_workspace(short_name,
                          version, "temp_workspace").push("src").push(fmt!("%s-%s",
                                                            short_name.to_str(),
                                                            version.to_str()));

    debug!("Created %s and does it exist? %?", package_dir.to_str(),
          os::path_is_dir(&package_dir));
    // Create main, lib, test, and bench files
    debug!("mk_workspace: creating %s", package_dir.to_str());
    assert!(os::mkdir_recursive(&package_dir, U_RWX));
    debug!("Created %s and does it exist? %?", package_dir.to_str(),
          os::path_is_dir(&package_dir));
    // Create main, lib, test, and bench files

    writeFile(&package_dir.push("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&package_dir.push("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&package_dir.push("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&package_dir.push("bench.rs"),
              "#[bench] pub fn f() { (); }");
    package_dir
}

fn run_git(args: &[~str], env: Option<~[(~str, ~str)]>, cwd: &Path, err_msg: &str) {
    let cwd = (*cwd).clone();
    let mut prog = run::Process::new("git", args, run::ProcessOptions {
        env: env,
        dir: Some(&cwd),
        in_fd: None,
        out_fd: None,
        err_fd: None
    });
    let rslt = prog.finish_with_output();
    if rslt.status != 0 {
        fail!("%s [git returned %?, output = %s, error = %s]", err_msg,
           rslt.status, str::from_bytes(rslt.output), str::from_bytes(rslt.error));
    }
}

/// Should create an empty git repo in p, relative to the tmp dir, and return the new
/// absolute path
fn init_git_repo(p: &Path) -> Path {
    assert!(!p.is_absolute());
    let tmp = mkdtemp(&os::tmpdir(), "git_local").expect("couldn't create temp dir");
    let work_dir = tmp.push_rel(p);
    let work_dir_for_opts = work_dir.clone();
    assert!(os::mkdir_recursive(&work_dir, U_RWX));
    debug!("Running: git init in %s", work_dir.to_str());
    let ws = work_dir.to_str();
    run_git([~"init"], None, &work_dir_for_opts,
        fmt!("Couldn't initialize git repository in %s", ws));
    // Add stuff to the dir so that git tag succeeds
    writeFile(&work_dir.push("README"), "");
    run_git([~"add", ~"README"], None, &work_dir_for_opts, fmt!("Couldn't add in %s", ws));
    git_commit(&work_dir_for_opts, ~"whatever");
    tmp
}

fn add_all_and_commit(repo: &Path) {
    git_add_all(repo);
    git_commit(repo, ~"floop");
}

fn git_commit(repo: &Path, msg: ~str) {
    run_git([~"commit", ~"--author=tester <test@mozilla.com>", ~"-m", msg],
            None, repo, fmt!("Couldn't commit in %s", repo.to_str()));
}

fn git_add_all(repo: &Path) {
    run_git([~"add", ~"-A"], None, repo, fmt!("Couldn't add all files in %s", repo.to_str()));
}

fn add_git_tag(repo: &Path, tag: ~str) {
    assert!(repo.is_absolute());
    git_add_all(repo);
    git_commit(repo, ~"whatever");
    run_git([~"tag", tag.clone()], None, repo,
            fmt!("Couldn't add git tag %s in %s", tag, repo.to_str()));
}

fn is_rwx(p: &Path) -> bool {
    use std::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};

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
    // Infer the sysroot from the exe name and pray that it's right.
    // (Did I mention it was a gross hack?)
    let self_path = os::self_exe_path().expect("Couldn't get self_exe path");
    self_path.pop()
}

// Returns the path to rustpkg
fn rustpkg_exec() -> Path {
    // Ugh
    let first_try = test_sysroot().push("lib").push("rustc")
        .push(host_triple()).push("bin").push("rustpkg");
    if is_executable(&first_try) {
        first_try
    }
    else {
        let second_try = test_sysroot().push("bin").push("rustpkg");
        if is_executable(&second_try) {
            second_try
        }
        else {
            fail!("in rustpkg test, can't find an installed rustpkg");
        }
    }
}

fn command_line_test(args: &[~str], cwd: &Path) -> ProcessOutput {
    command_line_test_with_env(args, cwd, None)
}

/// Runs `rustpkg` (based on the directory that this executable was
/// invoked from) with the given arguments, in the given working directory.
/// Returns the process's output.
fn command_line_test_with_env(args: &[~str], cwd: &Path, env: Option<~[(~str, ~str)]>)
    -> ProcessOutput {
    let cmd = rustpkg_exec().to_str();
    debug!("cd %s; %s %s",
           cwd.to_str(), cmd, args.connect(" "));
    assert!(os::path_is_dir(&*cwd));
    let cwd = (*cwd).clone();
    let mut prog = run::Process::new(cmd, args, run::ProcessOptions {
        env: env.map(|e| e + os::env()),
        dir: Some(&cwd),
        in_fd: None,
        out_fd: None,
        err_fd: None
    });
    let output = prog.finish_with_output();
    debug!("Output from command %s with args %? was %s {%s}[%?]",
                    cmd, args, str::from_bytes(output.output),
                   str::from_bytes(output.error),
                   output.status);
/*
By the way, rustpkg *won't* return a nonzero exit code if it fails --
see #4547
So tests that use this need to check the existence of a file
to make sure the command succeeded
*/
    if output.status != 0 {
        fail!("Command %s %? failed with exit code %?; its output was {{{ %s }}}",
              cmd, args, output.status,
              str::from_bytes(output.output) + str::from_bytes(output.error));
    }
    output
}

fn create_local_package(pkgid: &PkgId) -> Path {
    let parent_dir = mk_temp_workspace(&pkgid.path, &pkgid.version);
    debug!("Created empty package dir for %s, returning %s", pkgid.to_str(), parent_dir.to_str());
    parent_dir.pop().pop()
}

fn create_local_package_in(pkgid: &PkgId, pkgdir: &Path) -> Path {

    let package_dir = pkgdir.push("src").push(pkgid.to_str());

    // Create main, lib, test, and bench files
    assert!(os::mkdir_recursive(&package_dir, U_RWX));
    debug!("Created %s and does it exist? %?", package_dir.to_str(),
          os::path_is_dir(&package_dir));
    // Create main, lib, test, and bench files

    writeFile(&package_dir.push("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&package_dir.push("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&package_dir.push("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&package_dir.push("bench.rs"),
              "#[bench] pub fn f() { (); }");
    package_dir
}

fn create_local_package_with_test(pkgid: &PkgId) -> Path {
    debug!("Dry run -- would create package %s with test");
    create_local_package(pkgid) // Already has tests???
}

fn create_local_package_with_dep(pkgid: &PkgId, subord_pkgid: &PkgId) -> Path {
    let package_dir = create_local_package(pkgid);
    create_local_package_in(subord_pkgid, &package_dir);
    // Write a main.rs file into pkgid that references subord_pkgid
    writeFile(&package_dir.push("src").push(pkgid.to_str()).push("main.rs"),
              fmt!("extern mod %s;\nfn main() {}",
                   subord_pkgid.short_name));
    // Write a lib.rs file into subord_pkgid that has something in it
    writeFile(&package_dir.push("src").push(subord_pkgid.to_str()).push("lib.rs"),
              "pub fn f() {}");
    debug!("Dry run -- would create packages %s and %s in %s",
           pkgid.to_str(),
           subord_pkgid.to_str(),
           package_dir.to_str());
    package_dir
}

fn create_local_package_with_custom_build_hook(pkgid: &PkgId,
                                               custom_build_hook: &str) -> Path {
    debug!("Dry run -- would create package %s with custom build hook %s",
           pkgid.to_str(), custom_build_hook);
    create_local_package(pkgid)
    // actually write the pkg.rs with the custom build hook

}

fn assert_lib_exists(repo: &Path, short_name: &str, v: Version) {
    assert!(lib_exists(repo, short_name, v));
}

fn lib_exists(repo: &Path, short_name: &str, _v: Version) -> bool { // ??? version?
    debug!("assert_lib_exists: repo = %s, short_name = %s", repo.to_str(), short_name);
    let lib = installed_library_in_workspace(short_name, repo);
    debug!("assert_lib_exists: checking whether %? exists", lib);
    lib.is_some() && {
        let libname = lib.get_ref();
        os::path_exists(libname) && is_rwx(libname)
    }
}

fn assert_executable_exists(repo: &Path, short_name: &str) {
    assert!(executable_exists(repo, short_name));
}

fn executable_exists(repo: &Path, short_name: &str) -> bool {
    debug!("assert_executable_exists: repo = %s, short_name = %s", repo.to_str(), short_name);
    let exec = target_executable_in_workspace(&PkgId::new(short_name), repo);
    os::path_exists(&exec) && is_rwx(&exec)
}

fn assert_built_executable_exists(repo: &Path, short_name: &str) {
    assert!(built_executable_exists(repo, short_name));
}

fn built_executable_exists(repo: &Path, short_name: &str) -> bool {
    debug!("assert_built_executable_exists: repo = %s, short_name = %s", repo.to_str(), short_name);
    let exec = built_executable_in_workspace(&PkgId::new(short_name), repo);
    exec.is_some() && {
       let execname = exec.get_ref();
       os::path_exists(execname) && is_rwx(execname)
    }
}

fn assert_built_library_exists(repo: &Path, short_name: &str) {
    assert!(built_library_exists(repo, short_name));
}

fn built_library_exists(repo: &Path, short_name: &str) -> bool {
    debug!("assert_built_library_exists: repo = %s, short_name = %s", repo.to_str(), short_name);
    let lib = built_library_in_workspace(&PkgId::new(short_name), repo);
    lib.is_some() && {
        let libname = lib.get_ref();
        os::path_exists(libname) && is_rwx(libname)
    }
}

fn command_line_test_output(args: &[~str]) -> ~[~str] {
    let mut result = ~[];
    let p_output = command_line_test(args, &os::getcwd());
    let test_output = str::from_bytes(p_output.output);
    for s in test_output.split_iter('\n') {
        result.push(s.to_owned());
    }
    result
}

fn command_line_test_output_with_env(args: &[~str], env: ~[(~str, ~str)]) -> ~[~str] {
    let mut result = ~[];
    let p_output = command_line_test_with_env(args, &os::getcwd(), Some(env));
    let test_output = str::from_bytes(p_output.output);
    for s in test_output.split_iter('\n') {
        result.push(s.to_owned());
    }
    result
}

// assumes short_name and path are one and the same -- I should fix
fn lib_output_file_name(workspace: &Path, parent: &str, short_name: &str) -> Path {
    debug!("lib_output_file_name: given %s and parent %s and short name %s",
           workspace.to_str(), parent, short_name);
    library_in_workspace(&Path(short_name),
                         short_name,
                         Build,
                         workspace,
                         "build",
                         &NoVersion).expect("lib_output_file_name")
}

fn output_file_name(workspace: &Path, short_name: &str) -> Path {
    workspace.push(fmt!("%s%s", short_name, os::EXE_SUFFIX))
}

fn touch_source_file(workspace: &Path, pkgid: &PkgId) {
    use conditions::bad_path::cond;
    let pkg_src_dir = workspace.push("src").push(pkgid.to_str());
    let contents = os::list_dir_path(&pkg_src_dir);
    for p in contents.iter() {
        if p.filetype() == Some(".rs") {
            // should be able to do this w/o a process
            if run::process_output("touch", [p.to_str()]).status != 0 {
                let _ = cond.raise((pkg_src_dir.clone(), ~"Bad path"));
            }
            break;
        }
    }
}

/// Add a blank line at the end
fn frob_source_file(workspace: &Path, pkgid: &PkgId) {
    use conditions::bad_path::cond;
    let pkg_src_dir = workspace.push("src").push(pkgid.to_str());
    let contents = os::list_dir_path(&pkg_src_dir);
    let mut maybe_p = None;
    for p in contents.iter() {
        if p.filetype() == Some(".rs") {
            maybe_p = Some(p);
            break;
        }
    }
    match maybe_p {
        Some(p) => {
            let w = io::file_writer(p, &[io::Append]);
            match w {
                Err(s) => { let _ = cond.raise((p.clone(), fmt!("Bad path: %s", s))); }
                Ok(w)  => w.write_line("")
            }
        }
        None => fail!(fmt!("frob_source_file failed to find a source file in %s",
                           pkg_src_dir.to_str()))
    }
}

#[test]
fn test_make_dir_rwx() {
    let temp = &os::tmpdir();
    let dir = temp.push("quux");
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
    use path_util::installed_library_in_workspace;

    let sysroot = test_sysroot();
    debug!("sysroot = %s", sysroot.to_str());
    let ctxt = fake_ctxt(Some(@sysroot));
    let temp_pkg_id = fake_pkg();
    let temp_workspace = mk_temp_workspace(&temp_pkg_id.path, &NoVersion).pop().pop();
    debug!("temp_workspace = %s", temp_workspace.to_str());
    // should have test, bench, lib, and main
    ctxt.install(&temp_workspace, &temp_pkg_id);
    // Check that all files exist
    let exec = target_executable_in_workspace(&temp_pkg_id, &temp_workspace);
    debug!("exec = %s", exec.to_str());
    assert!(os::path_exists(&exec));
    assert!(is_rwx(&exec));

    let lib = installed_library_in_workspace(temp_pkg_id.short_name, &temp_workspace);
    debug!("lib = %?", lib);
    assert!(lib.map_default(false, |l| os::path_exists(l)));
    assert!(lib.map_default(false, |l| is_rwx(l)));

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
    use cond2 = conditions::not_a_workspace::cond;

    let ctxt = fake_ctxt(None);
    let pkgid = fake_pkg();
    let temp_workspace = mkdtemp(&os::tmpdir(), "test").expect("couldn't create temp dir");
    let mut error_occurred = false;
    let mut error1_occurred = false;
    let mut error2_occurred = false;
    do cond1.trap(|_| {
        error1_occurred = true;
    }).inside {
        do cond.trap(|_| {
            error_occurred = true;
            temp_workspace.clone()
        }).inside {
            do cond2.trap(|_| {
               error2_occurred = true;
               temp_workspace.clone()
            }).inside {
                 ctxt.install(&temp_workspace, &pkgid);
            }
        }
    }
    assert!(error_occurred && error1_occurred && error2_occurred);
}

// Tests above should (maybe) be converted to shell out to rustpkg, too
#[test]
fn test_install_git() {
    let sysroot = test_sysroot();
    debug!("sysroot = %s", sysroot.to_str());
    let temp_pkg_id = git_repo_pkg();
    let repo = init_git_repo(&temp_pkg_id.path);
    debug!("repo = %s", repo.to_str());
    let repo_subdir = repo.push("mockgithub.com").push("catamorphism").push("test-pkg");
    debug!("repo_subdir = %s", repo_subdir.to_str());

    writeFile(&repo_subdir.push("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&repo_subdir.push("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&repo_subdir.push("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&repo_subdir.push("bench.rs"),
              "#[bench] pub fn f() { (); }");
    add_git_tag(&repo_subdir, ~"0.1"); // this has the effect of committing the files

    debug!("test_install_git: calling rustpkg install %s in %s",
           temp_pkg_id.path.to_str(), repo.to_str());
    // should have test, bench, lib, and main
    command_line_test([~"install", temp_pkg_id.path.to_str()], &repo);
    let ws = repo.push(".rust");
    // Check that all files exist
    debug!("Checking for files in %s", ws.to_str());
    let exec = target_executable_in_workspace(&temp_pkg_id, &ws);
    debug!("exec = %s", exec.to_str());
    assert!(os::path_exists(&exec));
    assert!(is_rwx(&exec));
    let _built_lib =
        built_library_in_workspace(&temp_pkg_id,
                                   &ws).expect("test_install_git: built lib should exist");
    assert_lib_exists(&ws, temp_pkg_id.short_name, temp_pkg_id.version.clone());
    let built_test = built_test_in_workspace(&temp_pkg_id,
                         &ws).expect("test_install_git: built test should exist");
    assert!(os::path_exists(&built_test));
    let built_bench = built_bench_in_workspace(&temp_pkg_id,
                          &ws).expect("test_install_git: built bench should exist");
    assert!(os::path_exists(&built_bench));
    // And that the test and bench executables aren't installed
    let test = target_test_in_workspace(&temp_pkg_id, &ws);
    assert!(!os::path_exists(&test));
    debug!("test = %s", test.to_str());
    let bench = target_bench_in_workspace(&temp_pkg_id, &ws);
    debug!("bench = %s", bench.to_str());
    assert!(!os::path_exists(&bench));
}

#[test]
fn test_package_ids_must_be_relative_path_like() {
    use conditions::bad_pkg_id::cond;

    /*
    Okay:
    - One identifier, with no slashes
    - Several slash-delimited things, with no / at the root

    Not okay:
    - Empty string
    - Absolute path (as per os::is_absolute)

    */

    let whatever = PkgId::new("foo");

    assert_eq!(~"foo-0.1", whatever.to_str());
    assert!("github.com/catamorphism/test-pkg-0.1" ==
            PkgId::new("github.com/catamorphism/test-pkg").to_str());

    do cond.trap(|(p, e)| {
        assert!("" == p.to_str());
        assert!("0-length pkgid" == e);
        whatever.clone()
    }).inside {
        let x = PkgId::new("");
        assert_eq!(~"foo-0.1", x.to_str());
    }

    do cond.trap(|(p, e)| {
        assert_eq!(p.to_str(), os::make_absolute(&Path("foo/bar/quux")).to_str());
        assert!("absolute pkgid" == e);
        whatever.clone()
    }).inside {
        let z = PkgId::new(os::make_absolute(&Path("foo/bar/quux")).to_str());
        assert_eq!(~"foo-0.1", z.to_str());
    }

}

#[test]
fn test_package_version() {
    let local_path = "mockgithub.com/catamorphism/test_pkg_version";
    let repo = init_git_repo(&Path(local_path));
    let repo_subdir = repo.push("mockgithub.com").push("catamorphism").push("test_pkg_version");
    debug!("Writing files in: %s", repo_subdir.to_str());
    writeFile(&repo_subdir.push("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&repo_subdir.push("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&repo_subdir.push("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&repo_subdir.push("bench.rs"),
              "#[bench] pub fn f() { (); }");
    add_git_tag(&repo_subdir, ~"0.4");

    // It won't pick up the 0.4 version because the dir isn't in the RUST_PATH, but...
    let temp_pkg_id = PkgId::new("mockgithub.com/catamorphism/test_pkg_version");
    // This should look at the prefix, clone into a workspace, then build.
    command_line_test([~"install", ~"mockgithub.com/catamorphism/test_pkg_version"],
                      &repo);
    let ws = repo.push(".rust");
    // we can still match on the filename to make sure it contains the 0.4 version
    assert!(match built_library_in_workspace(&temp_pkg_id,
                                             &ws) {
        Some(p) => p.to_str().ends_with(fmt!("0.4%s", os::consts::DLL_SUFFIX)),
        None    => false
    });
    assert!(built_executable_in_workspace(&temp_pkg_id, &ws)
            == Some(ws.push("build").
                    push("mockgithub.com").
                    push("catamorphism").
                    push("test_pkg_version").
                    push("test_pkg_version")));
}

#[test]
fn test_package_request_version() {
    let local_path = "mockgithub.com/catamorphism/test_pkg_version";
    let repo = init_git_repo(&Path(local_path));
    let repo_subdir = repo.push("mockgithub.com").push("catamorphism").push("test_pkg_version");
    debug!("Writing files in: %s", repo_subdir.to_str());
    writeFile(&repo_subdir.push("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&repo_subdir.push("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&repo_subdir.push("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&repo_subdir.push("bench.rs"),
              "#[bench] pub fn f() { (); }");
    writeFile(&repo_subdir.push("version-0.3-file.txt"), "hi");
    add_git_tag(&repo_subdir, ~"0.3");
    writeFile(&repo_subdir.push("version-0.4-file.txt"), "hello");
    add_git_tag(&repo_subdir, ~"0.4");

    command_line_test([~"install", fmt!("%s#0.3", local_path)], &repo);

    assert!(match installed_library_in_workspace("test_pkg_version", &repo.push(".rust")) {
        Some(p) => {
            debug!("installed: %s", p.to_str());
            p.to_str().ends_with(fmt!("0.3%s", os::consts::DLL_SUFFIX))
        }
        None    => false
    });
    let temp_pkg_id = PkgId::new("mockgithub.com/catamorphism/test_pkg_version#0.3");
    assert!(target_executable_in_workspace(&temp_pkg_id, &repo.push(".rust"))
            == repo.push(".rust").push("bin").push("test_pkg_version"));

    assert!(os::path_exists(&repo.push(".rust").push("src")
                            .push("mockgithub.com").push("catamorphism")
                            .push("test_pkg_version-0.3")
                            .push("version-0.3-file.txt")));
    assert!(!os::path_exists(&repo.push(".rust").push("src")
                            .push("mockgithub.com").push("catamorphism")
                             .push("test_pkg_version-0.3")
                            .push("version-0.4-file.txt")));
}

#[test]
#[ignore (reason = "http-client not ported to rustpkg yet")]
fn rustpkg_install_url_2() {
    let temp_dir = mkdtemp(&os::tmpdir(), "rustpkg_install_url_2").expect("rustpkg_install_url_2");
    command_line_test([~"install", ~"github.com/mozilla-servo/rust-http-client"],
                     &temp_dir);
}

#[test]
fn rustpkg_library_target() {
    let foo_repo = init_git_repo(&Path("foo"));
    let package_dir = foo_repo.push("foo");

    debug!("Writing files in: %s", package_dir.to_str());
    writeFile(&package_dir.push("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&package_dir.push("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&package_dir.push("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&package_dir.push("bench.rs"),
              "#[bench] pub fn f() { (); }");

    add_git_tag(&package_dir, ~"1.0");
    command_line_test([~"install", ~"foo"], &foo_repo);
    assert_lib_exists(&foo_repo.push(".rust"), "foo", ExactRevision(~"1.0"));
}

#[test]
fn rustpkg_local_pkg() {
    let dir = create_local_package(&PkgId::new("foo"));
    command_line_test([~"install", ~"foo"], &dir);
    assert_executable_exists(&dir, "foo");
}

#[test]
#[ignore (reason = "test makes bogus assumptions about build directory layout: issue #8690")]
fn package_script_with_default_build() {
    let dir = create_local_package(&PkgId::new("fancy-lib"));
    debug!("dir = %s", dir.to_str());
    let source = test_sysroot().pop().pop().pop().push("src").push("librustpkg").
        push("testsuite").push("pass").push("src").push("fancy-lib").push("pkg.rs");
    debug!("package_script_with_default_build: %s", source.to_str());
    if !os::copy_file(&source,
                      &dir.push("src").push("fancy-lib-0.1").push("pkg.rs")) {
        fail!("Couldn't copy file");
    }
    command_line_test([~"install", ~"fancy-lib"], &dir);
    assert_lib_exists(&dir, "fancy-lib", NoVersion);
    assert!(os::path_exists(&dir.push("build").push("fancy-lib").push("generated.rs")));
}

#[test]
fn rustpkg_build_no_arg() {
    let tmp = mkdtemp(&os::tmpdir(), "rustpkg_build_no_arg").expect("rustpkg_build_no_arg failed")
              .push(".rust");
    let package_dir = tmp.push("src").push("foo");
    assert!(os::mkdir_recursive(&package_dir, U_RWX));

    writeFile(&package_dir.push("main.rs"),
              "fn main() { let _x = (); }");
    debug!("build_no_arg: dir = %s", package_dir.to_str());
    command_line_test([~"build"], &package_dir);
    assert_built_executable_exists(&tmp, "foo");
}

#[test]
fn rustpkg_install_no_arg() {
    let tmp = mkdtemp(&os::tmpdir(),
                      "rustpkg_install_no_arg").expect("rustpkg_install_no_arg failed")
              .push(".rust");
    let package_dir = tmp.push("src").push("foo");
    assert!(os::mkdir_recursive(&package_dir, U_RWX));
    writeFile(&package_dir.push("lib.rs"),
              "fn main() { let _x = (); }");
    debug!("install_no_arg: dir = %s", package_dir.to_str());
    command_line_test([~"install"], &package_dir);
    assert_lib_exists(&tmp, "foo", NoVersion);
}

#[test]
fn rustpkg_clean_no_arg() {
    let tmp = mkdtemp(&os::tmpdir(), "rustpkg_clean_no_arg").expect("rustpkg_clean_no_arg failed")
              .push(".rust");
    let package_dir = tmp.push("src").push("foo");
    assert!(os::mkdir_recursive(&package_dir, U_RWX));

    writeFile(&package_dir.push("main.rs"),
              "fn main() { let _x = (); }");
    debug!("clean_no_arg: dir = %s", package_dir.to_str());
    command_line_test([~"build"], &package_dir);
    assert_built_executable_exists(&tmp, "foo");
    command_line_test([~"clean"], &package_dir);
    assert!(!built_executable_in_workspace(&PkgId::new("foo"),
                &tmp).map_default(false, |m| { os::path_exists(m) }));
}

#[test]
fn rust_path_test() {
    let dir_for_path = mkdtemp(&os::tmpdir(), "more_rust").expect("rust_path_test failed");
    let dir = mk_workspace(&dir_for_path, &Path("foo"), &NoVersion);
    debug!("dir = %s", dir.to_str());
    writeFile(&dir.push("main.rs"), "fn main() { let _x = (); }");

    let cwd = os::getcwd();
    debug!("cwd = %s", cwd.to_str());
                                     // use command_line_test_with_env
    command_line_test_with_env([~"install", ~"foo"],
                               &cwd,
                               Some(~[(~"RUST_PATH", dir_for_path.to_str())]));
    assert_executable_exists(&dir_for_path, "foo");
}

#[test]
fn rust_path_contents() {
    use std::unstable::change_dir_locked;

    let dir = mkdtemp(&os::tmpdir(), "rust_path").expect("rust_path_contents failed");
    let abc = &dir.push("A").push("B").push("C");
    assert!(os::mkdir_recursive(&abc.push(".rust"), U_RWX));
    assert!(os::mkdir_recursive(&abc.pop().push(".rust"), U_RWX));
    assert!(os::mkdir_recursive(&abc.pop().pop().push(".rust"), U_RWX));
    assert!(do change_dir_locked(&dir.push("A").push("B").push("C")) {
        let p = rust_path();
        let cwd = os::getcwd().push(".rust");
        let parent = cwd.pop().pop().push(".rust");
        let grandparent = cwd.pop().pop().pop().push(".rust");
        assert!(p.contains(&cwd));
        assert!(p.contains(&parent));
        assert!(p.contains(&grandparent));
        for a_path in p.iter() {
            assert!(!a_path.components.is_empty());
        }
    });
}

#[test]
fn rust_path_parse() {
    os::setenv("RUST_PATH", "/a/b/c:/d/e/f:/g/h/i");
    let paths = rust_path();
    assert!(paths.contains(&Path("/g/h/i")));
    assert!(paths.contains(&Path("/d/e/f")));
    assert!(paths.contains(&Path("/a/b/c")));
    os::unsetenv("RUST_PATH");
}

#[test]
fn test_list() {
    let dir = mkdtemp(&os::tmpdir(), "test_list").expect("test_list failed");
    let foo = PkgId::new("foo");
    create_local_package_in(&foo, &dir);
    let bar = PkgId::new("bar");
    create_local_package_in(&bar, &dir);
    let quux = PkgId::new("quux");
    create_local_package_in(&quux, &dir);

// list doesn't output very much right now...
    command_line_test([~"install", ~"foo"], &dir);
    let env_arg = ~[(~"RUST_PATH", dir.to_str())];
    debug!("RUST_PATH = %s", dir.to_str());
    let list_output = command_line_test_output_with_env([~"list"], env_arg.clone());
    assert!(list_output.iter().any(|x| x.starts_with("foo")));

    command_line_test([~"install", ~"bar"], &dir);
    let list_output = command_line_test_output_with_env([~"list"], env_arg.clone());
    assert!(list_output.iter().any(|x| x.starts_with("foo")));
    assert!(list_output.iter().any(|x| x.starts_with("bar")));

    command_line_test([~"install", ~"quux"], &dir);
    let list_output = command_line_test_output_with_env([~"list"], env_arg);
    assert!(list_output.iter().any(|x| x.starts_with("foo")));
    assert!(list_output.iter().any(|x| x.starts_with("bar")));
    assert!(list_output.iter().any(|x| x.starts_with("quux")));
}

#[test]
fn install_remove() {
    let dir = mkdtemp(&os::tmpdir(), "install_remove").expect("install_remove");
    let foo = PkgId::new("foo");
    let bar = PkgId::new("bar");
    let quux = PkgId::new("quux");
    create_local_package_in(&foo, &dir);
    create_local_package_in(&bar, &dir);
    create_local_package_in(&quux, &dir);
    let rust_path_to_use = ~[(~"RUST_PATH", dir.to_str())];
    command_line_test([~"install", ~"foo"], &dir);
    command_line_test([~"install", ~"bar"], &dir);
    command_line_test([~"install", ~"quux"], &dir);
    let list_output = command_line_test_output_with_env([~"list"], rust_path_to_use.clone());
    assert!(list_output.iter().any(|x| x.starts_with("foo")));
    assert!(list_output.iter().any(|x| x.starts_with("bar")));
    assert!(list_output.iter().any(|x| x.starts_with("quux")));
    command_line_test([~"uninstall", ~"foo"], &dir);
    let list_output = command_line_test_output_with_env([~"list"], rust_path_to_use.clone());
    assert!(!list_output.iter().any(|x| x.starts_with("foo")));
    assert!(list_output.iter().any(|x| x.starts_with("bar")));
    assert!(list_output.iter().any(|x| x.starts_with("quux")));
}

#[test]
fn install_check_duplicates() {
    // should check that we don't install two packages with the same full name *and* version
    // ("Is already installed -- doing nothing")
    // check invariant that there are no dups in the pkg database
    let dir = mkdtemp(&os::tmpdir(), "install_remove").expect("install_remove");
    let foo = PkgId::new("foo");
    create_local_package_in(&foo, &dir);

    command_line_test([~"install", ~"foo"], &dir);
    command_line_test([~"install", ~"foo"], &dir);
    let mut contents = ~[];
    let check_dups = |p: &PkgId| {
        if contents.contains(p) {
            fail!("package %s appears in `list` output more than once", p.path.to_str());
        }
        else {
            contents.push((*p).clone());
        }
        false
    };
    list_installed_packages(check_dups);
}

#[test]
#[ignore(reason = "Workcache not yet implemented -- see #7075")]
fn no_rebuilding() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    command_line_test([~"build", ~"foo"], &workspace);
    let date = datestamp(&built_library_in_workspace(&p_id,
                                                    &workspace).expect("no_rebuilding"));
    command_line_test([~"build", ~"foo"], &workspace);
    let newdate = datestamp(&built_library_in_workspace(&p_id,
                                                       &workspace).expect("no_rebuilding (2)"));
    assert_eq!(date, newdate);
}

#[test]
#[ignore(reason = "Workcache not yet implemented -- see #7075")]
fn no_rebuilding_dep() {
    let p_id = PkgId::new("foo");
    let dep_id = PkgId::new("bar");
    let workspace = create_local_package_with_dep(&p_id, &dep_id);
    command_line_test([~"build", ~"foo"], &workspace);
    let bar_date = datestamp(&lib_output_file_name(&workspace,
                                                  ".rust",
                                                  "bar"));
    let foo_date = datestamp(&output_file_name(&workspace, "foo"));
    assert!(bar_date < foo_date);
}

// n.b. The following two tests are ignored; they worked "accidentally" before,
// when the behavior was "always rebuild libraries" (now it's "never rebuild
// libraries if they already exist"). They can be un-ignored once #7075 is done.
#[test]
#[ignore(reason = "Workcache not yet implemented -- see #7075")]
fn do_rebuild_dep_dates_change() {
    let p_id = PkgId::new("foo");
    let dep_id = PkgId::new("bar");
    let workspace = create_local_package_with_dep(&p_id, &dep_id);
    command_line_test([~"build", ~"foo"], &workspace);
    let bar_lib_name = lib_output_file_name(&workspace, "build", "bar");
    let bar_date = datestamp(&bar_lib_name);
    debug!("Datestamp on %s is %?", bar_lib_name.to_str(), bar_date);
    touch_source_file(&workspace, &dep_id);
    command_line_test([~"build", ~"foo"], &workspace);
    let new_bar_date = datestamp(&bar_lib_name);
    debug!("Datestamp on %s is %?", bar_lib_name.to_str(), new_bar_date);
    assert!(new_bar_date > bar_date);
}

#[test]
#[ignore(reason = "Workcache not yet implemented -- see #7075")]
fn do_rebuild_dep_only_contents_change() {
    let p_id = PkgId::new("foo");
    let dep_id = PkgId::new("bar");
    let workspace = create_local_package_with_dep(&p_id, &dep_id);
    command_line_test([~"build", ~"foo"], &workspace);
    let bar_date = datestamp(&lib_output_file_name(&workspace, "build", "bar"));
    frob_source_file(&workspace, &dep_id);
    // should adjust the datestamp
    command_line_test([~"build", ~"foo"], &workspace);
    let new_bar_date = datestamp(&lib_output_file_name(&workspace, "build", "bar"));
    assert!(new_bar_date > bar_date);
}

#[test]
fn test_versions() {
    let workspace = create_local_package(&PkgId::new("foo#0.1"));
    create_local_package(&PkgId::new("foo#0.2"));
    command_line_test([~"install", ~"foo#0.1"], &workspace);
    let output = command_line_test_output([~"list"]);
    // make sure output includes versions
    assert!(!output.iter().any(|x| x == &~"foo#0.2"));
}

#[test]
#[ignore(reason = "do not yet implemented")]
fn test_build_hooks() {
    let workspace = create_local_package_with_custom_build_hook(&PkgId::new("foo"),
                                                                "frob");
    command_line_test([~"do", ~"foo", ~"frob"], &workspace);
}


#[test]
#[ignore(reason = "info not yet implemented")]
fn test_info() {
    let expected_info = ~"package foo"; // fill in
    let workspace = create_local_package(&PkgId::new("foo"));
    let output = command_line_test([~"info", ~"foo"], &workspace);
    assert_eq!(str::from_bytes(output.output), expected_info);
}

#[test]
#[ignore(reason = "test not yet implemented")]
fn test_rustpkg_test() {
    let expected_results = ~"1 out of 1 tests passed"; // fill in
    let workspace = create_local_package_with_test(&PkgId::new("foo"));
    let output = command_line_test([~"test", ~"foo"], &workspace);
    assert_eq!(str::from_bytes(output.output), expected_results);
}

#[test]
#[ignore(reason = "test not yet implemented")]
fn test_uninstall() {
    let workspace = create_local_package(&PkgId::new("foo"));
    let _output = command_line_test([~"info", ~"foo"], &workspace);
    command_line_test([~"uninstall", ~"foo"], &workspace);
    let output = command_line_test([~"list"], &workspace);
    assert!(!str::from_bytes(output.output).contains("foo"));
}

#[test]
fn test_non_numeric_tag() {
    let temp_pkg_id = git_repo_pkg();
    let repo = init_git_repo(&temp_pkg_id.path);
    let repo_subdir = repo.push("mockgithub.com").push("catamorphism").push("test-pkg");
    writeFile(&repo_subdir.push("foo"), "foo");
    writeFile(&repo_subdir.push("lib.rs"),
              "pub fn f() { let _x = (); }");
    add_git_tag(&repo_subdir, ~"testbranch");
    writeFile(&repo_subdir.push("testbranch_only"), "hello");
    add_git_tag(&repo_subdir, ~"another_tag");
    writeFile(&repo_subdir.push("not_on_testbranch_only"), "bye bye");
    add_all_and_commit(&repo_subdir);

    command_line_test([~"install", fmt!("%s#testbranch", temp_pkg_id.path.to_str())], &repo);
    let file1 = repo.push_many(["mockgithub.com", "catamorphism",
                                "test-pkg", "testbranch_only"]);
    let file2 = repo.push_many(["mockgithub.com", "catamorphism", "test-pkg",
                                "master_only"]);
    assert!(os::path_exists(&file1));
    assert!(!os::path_exists(&file2));
}

#[test]
fn test_extern_mod() {
    let dir = mkdtemp(&os::tmpdir(), "test_extern_mod").expect("test_extern_mod");
    let main_file = dir.push("main.rs");
    let lib_depend_dir = mkdtemp(&os::tmpdir(), "foo").expect("test_extern_mod");
    let aux_dir = lib_depend_dir.push_many(["src", "mockgithub.com", "catamorphism", "test_pkg"]);
    assert!(os::mkdir_recursive(&aux_dir, U_RWX));
    let aux_pkg_file = aux_dir.push("lib.rs");

    writeFile(&aux_pkg_file, "pub mod bar { pub fn assert_true() {  assert!(true); } }\n");
    assert!(os::path_exists(&aux_pkg_file));

    writeFile(&main_file,
              "extern mod test = \"mockgithub.com/catamorphism/test_pkg\";\nuse test::bar;\
               fn main() { bar::assert_true(); }\n");

    command_line_test([~"install", ~"mockgithub.com/catamorphism/test_pkg"], &lib_depend_dir);

    let exec_file = dir.push("out");
    // Be sure to extend the existing environment
    let env = Some([(~"RUST_PATH", lib_depend_dir.to_str())] + os::env());
    let rustpkg_exec = rustpkg_exec();
    let rustc = rustpkg_exec.with_filename("rustc");
    debug!("RUST_PATH=%s %s %s \n --sysroot %s -o %s",
                     lib_depend_dir.to_str(),
                     rustc.to_str(),
                     main_file.to_str(),
                     test_sysroot().to_str(),
                     exec_file.to_str());

    let mut prog = run::Process::new(rustc.to_str(), [main_file.to_str(),
                                                      ~"--sysroot", test_sysroot().to_str(),
                                               ~"-o", exec_file.to_str()],
                                     run::ProcessOptions {
        env: env,
        dir: Some(&dir),
        in_fd: None,
        out_fd: None,
        err_fd: None
    });
    let outp = prog.finish_with_output();
    if outp.status != 0 {
        fail!("output was %s, error was %s",
              str::from_bytes(outp.output),
              str::from_bytes(outp.error));
    }
    assert!(os::path_exists(&exec_file) && is_executable(&exec_file));
}

#[test]
fn test_import_rustpkg() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    writeFile(&workspace.push("src").push("foo-0.1").push("pkg.rs"),
              "extern mod rustpkg; fn main() {}");
    command_line_test([~"build", ~"foo"], &workspace);
    debug!("workspace = %s", workspace.to_str());
    assert!(os::path_exists(&workspace.push("build").push("foo").push(fmt!("pkg%s",
        os::EXE_SUFFIX))));
}

#[test]
fn test_macro_pkg_script() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    writeFile(&workspace.push("src").push("foo-0.1").push("pkg.rs"),
              "extern mod rustpkg; fn main() { debug!(\"Hi\"); }");
    command_line_test([~"build", ~"foo"], &workspace);
    debug!("workspace = %s", workspace.to_str());
    assert!(os::path_exists(&workspace.push("build").push("foo").push(fmt!("pkg%s",
        os::EXE_SUFFIX))));
}

#[test]
fn multiple_workspaces() {
// Make a package foo; build/install in directory A
// Copy the exact same package into directory B and install it
// Set the RUST_PATH to A:B
// Make a third package that uses foo, make sure we can build/install it
    let a_loc = mk_temp_workspace(&Path("foo"), &NoVersion).pop().pop();
    let b_loc = mk_temp_workspace(&Path("foo"), &NoVersion).pop().pop();
    debug!("Trying to install foo in %s", a_loc.to_str());
    command_line_test([~"install", ~"foo"], &a_loc);
    debug!("Trying to install foo in %s", b_loc.to_str());
    command_line_test([~"install", ~"foo"], &b_loc);
    let env = Some(~[(~"RUST_PATH", fmt!("%s:%s", a_loc.to_str(), b_loc.to_str()))]);
    let c_loc = create_local_package_with_dep(&PkgId::new("bar"), &PkgId::new("foo"));
    command_line_test_with_env([~"install", ~"bar"], &c_loc, env);
}

fn rust_path_hack_test(hack_flag: bool) {
/*
      Make a workspace containing a pkg foo [A]
      Make a second, empty workspace        [B]
      Set RUST_PATH to B:A
      rustpkg install foo
      make sure built files for foo are in B
      make sure nothing gets built into A or A/../build[lib,bin]
*/
   let p_id = PkgId::new("foo");
   let workspace = create_local_package(&p_id);
   let dest_workspace = mk_empty_workspace(&Path("bar"), &NoVersion, "dest_workspace");
   let rust_path = Some(~[(~"RUST_PATH",
       fmt!("%s:%s", dest_workspace.to_str(), workspace.push_many(["src", "foo-0.1"]).to_str()))]);
   debug!("declare -x RUST_PATH=%s:%s",
       dest_workspace.to_str(), workspace.push_many(["src", "foo-0.1"]).to_str());
   command_line_test_with_env(~[~"install"] + if hack_flag { ~[~"--rust-path-hack"] } else { ~[] } +
                               ~[~"foo"], &dest_workspace, rust_path);
   assert_lib_exists(&dest_workspace, "foo", NoVersion);
   assert_executable_exists(&dest_workspace, "foo");
   assert_built_library_exists(&dest_workspace, "foo");
   assert_built_executable_exists(&dest_workspace, "foo");
   assert!(!lib_exists(&workspace, "foo", NoVersion));
   assert!(!executable_exists(&workspace, "foo"));
   assert!(!built_library_exists(&workspace, "foo"));
   assert!(!built_executable_exists(&workspace, "foo"));
}

#[test]
fn test_rust_path_can_contain_package_dirs_with_flag() {
/*
   Test that the temporary hack added for bootstrapping Servo builds
   works. That is: if you add $FOO/src/some_pkg to the RUST_PATH,
   it will find the sources in some_pkg, build them, and install them
   into the first entry in the RUST_PATH.

   When the hack is removed, we should change this to a should_fail test.
*/
   rust_path_hack_test(true);
}

#[test]
#[should_fail]
fn test_rust_path_can_contain_package_dirs_without_flag() {
   rust_path_hack_test(false);
}

#[test]
fn rust_path_hack_cwd() {
   // Same as rust_path_hack_test, but the CWD is the dir to build out of
   let cwd = mkdtemp(&os::tmpdir(), "pkg_files").expect("rust_path_hack_cwd");
   writeFile(&cwd.push("lib.rs"), "pub fn f() { }");

   let dest_workspace = mk_empty_workspace(&Path("bar"), &NoVersion, "dest_workspace");
   let rust_path = Some(~[(~"RUST_PATH", dest_workspace.to_str())]);
   debug!("declare -x RUST_PATH=%s", dest_workspace.to_str());
   command_line_test_with_env([~"install", ~"--rust-path-hack", ~"foo"], &cwd, rust_path);
   debug!("Checking that foo exists in %s", dest_workspace.to_str());
   assert_lib_exists(&dest_workspace, "foo", NoVersion);
   assert_built_library_exists(&dest_workspace, "foo");
   assert!(!lib_exists(&cwd, "foo", NoVersion));
   assert!(!built_library_exists(&cwd, "foo"));
}

#[test]
fn rust_path_hack_multi_path() {
   // Same as rust_path_hack_test, but with a more complex package ID
   let cwd = mkdtemp(&os::tmpdir(), "pkg_files").expect("rust_path_hack_cwd");
   let subdir = cwd.push_many([~"foo", ~"bar", ~"quux"]);
   assert!(os::mkdir_recursive(&subdir, U_RWX));
   writeFile(&subdir.push("lib.rs"), "pub fn f() { }");
   let name = ~"foo/bar/quux";

   let dest_workspace = mk_empty_workspace(&Path("bar"), &NoVersion, "dest_workspace");
   let rust_path = Some(~[(~"RUST_PATH", dest_workspace.to_str())]);
   debug!("declare -x RUST_PATH=%s", dest_workspace.to_str());
   command_line_test_with_env([~"install", ~"--rust-path-hack", name.clone()], &subdir, rust_path);
   debug!("Checking that %s exists in %s", name, dest_workspace.to_str());
   assert_lib_exists(&dest_workspace, "quux", NoVersion);
   assert_built_library_exists(&dest_workspace, name);
   assert!(!lib_exists(&subdir, "quux", NoVersion));
   assert!(!built_library_exists(&subdir, name));
}

#[test]
fn rust_path_hack_install_no_arg() {
   // Same as rust_path_hack_cwd, but making rustpkg infer the pkg id
   let cwd = mkdtemp(&os::tmpdir(), "pkg_files").expect("rust_path_hack_install_no_arg");
   let source_dir = cwd.push("foo");
   assert!(make_dir_rwx(&source_dir));
   writeFile(&source_dir.push("lib.rs"), "pub fn f() { }");

   let dest_workspace = mk_empty_workspace(&Path("bar"), &NoVersion, "dest_workspace");
   let rust_path = Some(~[(~"RUST_PATH", dest_workspace.to_str())]);
   debug!("declare -x RUST_PATH=%s", dest_workspace.to_str());
   command_line_test_with_env([~"install", ~"--rust-path-hack"], &source_dir, rust_path);
   debug!("Checking that foo exists in %s", dest_workspace.to_str());
   assert_lib_exists(&dest_workspace, "foo", NoVersion);
   assert_built_library_exists(&dest_workspace, "foo");
   assert!(!lib_exists(&source_dir, "foo", NoVersion));
   assert!(!built_library_exists(&cwd, "foo"));
}

#[test]
fn rust_path_hack_build_no_arg() {
   // Same as rust_path_hack_install_no_arg, but building instead of installing
   let cwd = mkdtemp(&os::tmpdir(), "pkg_files").expect("rust_path_hack_build_no_arg");
   let source_dir = cwd.push("foo");
   assert!(make_dir_rwx(&source_dir));
   writeFile(&source_dir.push("lib.rs"), "pub fn f() { }");

   let dest_workspace = mk_empty_workspace(&Path("bar"), &NoVersion, "dest_workspace");
   let rust_path = Some(~[(~"RUST_PATH", dest_workspace.to_str())]);
   debug!("declare -x RUST_PATH=%s", dest_workspace.to_str());
   command_line_test_with_env([~"build", ~"--rust-path-hack"], &source_dir, rust_path);
   debug!("Checking that foo exists in %s", dest_workspace.to_str());
   assert_built_library_exists(&dest_workspace, "foo");
   assert!(!built_library_exists(&source_dir, "foo"));
}

#[test]
#[ignore (reason = "#7402 not yet implemented")]
fn rust_path_install_target() {
    let dir_for_path = mkdtemp(&os::tmpdir(),
        "source_workspace").expect("rust_path_install_target failed");
    let dir = mk_workspace(&dir_for_path, &Path("foo"), &NoVersion);
    debug!("dir = %s", dir.to_str());
    writeFile(&dir.push("main.rs"), "fn main() { let _x = (); }");
    let dir_to_install_to = mkdtemp(&os::tmpdir(),
        "dest_workspace").expect("rust_path_install_target failed");
    let dir = dir.pop().pop();

    let rust_path = Some(~[(~"RUST_PATH", fmt!("%s:%s", dir_to_install_to.to_str(),
                                               dir.to_str()))]);
    let cwd = os::getcwd();

    debug!("RUST_PATH=%s:%s", dir_to_install_to.to_str(), dir.to_str());
    command_line_test_with_env([~"install", ~"foo"],
                               &cwd,
                               rust_path);

    assert_executable_exists(&dir_to_install_to, "foo");

}


/// Returns true if p exists and is executable
fn is_executable(p: &Path) -> bool {
    use std::libc::consts::os::posix88::{S_IXUSR};

    match p.get_mode() {
        None => false,
        Some(mode) => mode & S_IXUSR as uint == S_IXUSR as uint
    }
}
