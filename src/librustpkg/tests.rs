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

use context::{BuildContext, Context, RustcFlags};
use std::{os, run, str, task};
use std::io;
use std::io::fs;
use std::io::File;
use extra::arc::Arc;
use extra::arc::RWArc;
use extra::tempfile::TempDir;
use extra::workcache;
use extra::workcache::{Database, Logger};
use extra::treemap::TreeMap;
use extra::getopts::groups::getopts;
use std::run::ProcessOutput;
use installed_packages::list_installed_packages;
use package_id::{PkgId};
use version::{ExactRevision, NoVersion, Version, Tagged};
use path_util::{target_executable_in_workspace, target_test_in_workspace,
               target_bench_in_workspace, make_dir_rwx,
               library_in_workspace, installed_library_in_workspace,
               built_bench_in_workspace, built_test_in_workspace,
               built_library_in_workspace, built_executable_in_workspace, target_build_dir,
               chmod_read_only, platform_library_name};
use rustc::back::link::get_cc_prog;
use rustc::metadata::filesearch::rust_path;
use rustc::driver::session;
use rustc::driver::driver::{build_session, build_session_options, host_triple, optgroups};
use syntax::diagnostic;
use target::*;
use package_source::PkgSrc;
use source_control::{CheckedOutSources, DirToUse, safe_git_clone};
use exit_codes::{BAD_FLAG_CODE, COPY_FAILED_CODE};

fn fake_ctxt(sysroot: Path, workspace: &Path) -> BuildContext {
    let context = workcache::Context::new(
        RWArc::new(Database::new(workspace.join("rustpkg_db.json"))),
        RWArc::new(Logger::new()),
        Arc::new(TreeMap::new()));
    BuildContext {
        workcache_context: context,
        context: Context {
            cfgs: ~[],
            rustc_flags: RustcFlags::default(),

            use_rust_path_hack: false,
            sysroot: sysroot
        }
    }
}

fn fake_pkg() -> PkgId {
    let sn = ~"bogus";
    PkgId {
        path: Path::new(sn.as_slice()),
        short_name: sn,
        version: NoVersion
    }
}

fn git_repo_pkg() -> PkgId {
    PkgId {
        path: Path::new("mockgithub.com/catamorphism/test-pkg"),
        short_name: ~"test-pkg",
        version: NoVersion
    }
}

fn git_repo_pkg_with_tag(a_tag: ~str) -> PkgId {
    PkgId {
        path: Path::new("mockgithub.com/catamorphism/test-pkg"),
        short_name: ~"test-pkg",
        version: Tagged(a_tag)
    }
}

fn writeFile(file_path: &Path, contents: &str) {
    let mut out = File::create(file_path);
    out.write(contents.as_bytes());
    out.write(['\n' as u8]);
}

fn mk_emptier_workspace(tag: &str) -> TempDir {
    let workspace = TempDir::new(tag).expect("couldn't create temp dir");
    let package_dir = workspace.path().join("src");
    fs::mkdir_recursive(&package_dir, io::UserRWX);
    workspace
}

fn mk_empty_workspace(short_name: &Path, version: &Version, tag: &str) -> TempDir {
    let workspace_dir = TempDir::new(tag).expect("couldn't create temp dir");
    mk_workspace(workspace_dir.path(), short_name, version);
    workspace_dir
}

fn mk_workspace(workspace: &Path, short_name: &Path, version: &Version) -> Path {
    // include version number in directory name
    // FIXME (#9639): This needs to handle non-utf8 paths
    let package_dir = workspace.join_many([~"src", format!("{}-{}",
                                           short_name.as_str().unwrap(), version.to_str())]);
    fs::mkdir_recursive(&package_dir, io::UserRWX);
    package_dir
}

fn mk_temp_workspace(short_name: &Path, version: &Version) -> (TempDir, Path) {
    let workspace_dir = mk_empty_workspace(short_name, version, "temp_workspace");
    // FIXME (#9639): This needs to handle non-utf8 paths
    let package_dir = workspace_dir.path().join_many([~"src",
                                                      format!("{}-{}",
                                                              short_name.as_str().unwrap(),
                                                              version.to_str())]);

    debug!("Created {} and does it exist? {:?}", package_dir.display(),
           package_dir.is_dir());
    // Create main, lib, test, and bench files
    debug!("mk_workspace: creating {}", package_dir.display());
    fs::mkdir_recursive(&package_dir, io::UserRWX);
    debug!("Created {} and does it exist? {:?}", package_dir.display(),
           package_dir.is_dir());
    // Create main, lib, test, and bench files

    writeFile(&package_dir.join("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&package_dir.join("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&package_dir.join("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&package_dir.join("bench.rs"),
              "#[bench] pub fn f() { (); }");
    (workspace_dir, package_dir)
}

fn run_git(args: &[~str], env: Option<~[(~str, ~str)]>, cwd: &Path, err_msg: &str) {
    let cwd = (*cwd).clone();
    let mut prog = run::Process::new("git", args, run::ProcessOptions {
        env: env,
        dir: Some(&cwd),
        in_fd: None,
        out_fd: None,
        err_fd: None
    }).expect("failed to exec `git`");
    let rslt = prog.finish_with_output();
    if !rslt.status.success() {
        fail!("{} [git returned {:?}, output = {}, error = {}]", err_msg,
           rslt.status, str::from_utf8(rslt.output), str::from_utf8(rslt.error));
    }
}

/// Should create an empty git repo in p, relative to the tmp dir, and return the new
/// absolute path
fn init_git_repo(p: &Path) -> TempDir {
    assert!(p.is_relative());
    let tmp = TempDir::new("git_local").expect("couldn't create temp dir");
    let work_dir = tmp.path().join(p);
    let work_dir_for_opts = work_dir.clone();
    fs::mkdir_recursive(&work_dir, io::UserRWX);
    debug!("Running: git init in {}", work_dir.display());
    run_git([~"init"], None, &work_dir_for_opts,
        format!("Couldn't initialize git repository in {}", work_dir.display()));
    // Add stuff to the dir so that git tag succeeds
    writeFile(&work_dir.join("README"), "");
    run_git([~"add", ~"README"], None, &work_dir_for_opts, format!("Couldn't add in {}",
                                                                work_dir.display()));
    git_commit(&work_dir_for_opts, ~"whatever");
    tmp
}

fn add_all_and_commit(repo: &Path) {
    git_add_all(repo);
    git_commit(repo, ~"floop");
}

fn git_commit(repo: &Path, msg: ~str) {
    run_git([~"commit", ~"--author=tester <test@mozilla.com>", ~"-m", msg],
            None, repo, format!("Couldn't commit in {}", repo.display()));
}

fn git_add_all(repo: &Path) {
    run_git([~"add", ~"-A"], None, repo, format!("Couldn't add all files in {}", repo.display()));
}

fn add_git_tag(repo: &Path, tag: ~str) {
    assert!(repo.is_absolute());
    git_add_all(repo);
    git_commit(repo, ~"whatever");
    run_git([~"tag", tag.clone()], None, repo,
            format!("Couldn't add git tag {} in {}", tag, repo.display()));
}

fn is_rwx(p: &Path) -> bool {
    if !p.exists() { return false }
    p.stat().perm & io::UserRWX == io::UserRWX
}

fn is_read_only(p: &Path) -> bool {
    if !p.exists() { return false }
    p.stat().perm & io::UserRWX == io::UserRead
}

fn test_sysroot() -> Path {
    // Totally gross hack but it's just for test cases.
    // Infer the sysroot from the exe name and pray that it's right.
    // (Did I mention it was a gross hack?)
    let mut self_path = os::self_exe_path().expect("Couldn't get self_exe path");
    self_path.pop();
    self_path
}

// Returns the path to rustpkg
fn rustpkg_exec() -> Path {
    // Ugh
    let first_try = test_sysroot().join_many(
        [~"lib", ~"rustc", host_triple(), ~"bin", ~"rustpkg"]);
    if is_executable(&first_try) {
        first_try
    }
    else {
        let second_try = test_sysroot().join_many(["bin", "rustpkg"]);
        if is_executable(&second_try) {
            second_try
        }
        else {
            fail!("in rustpkg test, can't find an installed rustpkg");
        }
    }
}

fn command_line_test(args: &[~str], cwd: &Path) -> ProcessOutput {
    match command_line_test_with_env(args, cwd, None) {
        Success(r) => r,
        Fail(error) => fail!("Command line test failed with error {}",
                             error.status)
    }
}

fn command_line_test_partial(args: &[~str], cwd: &Path) -> ProcessResult {
    command_line_test_with_env(args, cwd, None)
}

fn command_line_test_expect_fail(args: &[~str],
                                 cwd: &Path,
                                 env: Option<~[(~str, ~str)]>,
                                 expected_exitcode: int) {
    match command_line_test_with_env(args, cwd, env) {
        Success(_) => fail!("Should have failed with {}, but it succeeded", expected_exitcode),
        Fail(ref error) if error.status.matches_exit_status(expected_exitcode) => (), // ok
        Fail(other) => fail!("Expected to fail with {}, but failed with {} instead",
                              expected_exitcode, other.status)
    }
}

enum ProcessResult {
    Success(ProcessOutput),
    Fail(ProcessOutput)
}

/// Runs `rustpkg` (based on the directory that this executable was
/// invoked from) with the given arguments, in the given working directory.
/// Returns the process's output.
fn command_line_test_with_env(args: &[~str], cwd: &Path, env: Option<~[(~str, ~str)]>)
    -> ProcessResult {
    // FIXME (#9639): This needs to handle non-utf8 paths
    let exec_path = rustpkg_exec();
    let cmd = exec_path.as_str().unwrap().to_owned();
    let env_str = match env {
        Some(ref pairs) => pairs.map(|&(ref k, ref v)| { format!("{}={}", *k, *v) }).connect(","),
        None        => ~""
    };
    debug!("{} cd {}; {} {}", env_str, cwd.display(), cmd, args.connect(" "));
    assert!(cwd.is_dir());
    let cwd = (*cwd).clone();
    let mut prog = run::Process::new(cmd, args, run::ProcessOptions {
        env: env.map(|e| e + os::env()),
        dir: Some(&cwd),
        in_fd: None,
        out_fd: None,
        err_fd: None
    }).expect(format!("failed to exec `{}`", cmd));
    let output = prog.finish_with_output();
    debug!("Output from command {} with args {:?} was {} \\{{}\\}[{:?}]",
           cmd, args, str::from_utf8(output.output),
           str::from_utf8(output.error),
           output.status);
    if !output.status.success() {
        debug!("Command {} {:?} failed with exit code {:?}; its output was --- {} {} ---",
              cmd, args, output.status,
              str::from_utf8(output.output), str::from_utf8(output.error));
        Fail(output)
    }
    else {
        Success(output)
    }
}

fn create_local_package(pkgid: &PkgId) -> TempDir {
    let (workspace, parent_dir) = mk_temp_workspace(&pkgid.path, &pkgid.version);
    debug!("Created empty package dir for {}, returning {}", pkgid.to_str(), parent_dir.display());
    workspace
}

fn create_local_package_in(pkgid: &PkgId, pkgdir: &Path) -> Path {

    let package_dir = pkgdir.join_many([~"src", pkgid.to_str()]);

    // Create main, lib, test, and bench files
    fs::mkdir_recursive(&package_dir, io::UserRWX);
    debug!("Created {} and does it exist? {:?}", package_dir.display(),
           package_dir.is_dir());
    // Create main, lib, test, and bench files

    writeFile(&package_dir.join("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&package_dir.join("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&package_dir.join("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&package_dir.join("bench.rs"),
              "#[bench] pub fn f() { (); }");
    package_dir
}

fn create_local_package_with_test(pkgid: &PkgId) -> TempDir {
    debug!("Dry run -- would create package {:?} with test", pkgid);
    create_local_package(pkgid) // Already has tests???
}

fn create_local_package_with_dep(pkgid: &PkgId, subord_pkgid: &PkgId) -> TempDir {
    let package_dir = create_local_package(pkgid);
    create_local_package_in(subord_pkgid, package_dir.path());
    // Write a main.rs file into pkgid that references subord_pkgid
    writeFile(&package_dir.path().join_many([~"src", pkgid.to_str(), ~"main.rs"]),
              format!("extern mod {};\nfn main() \\{\\}",
                   subord_pkgid.short_name));
    // Write a lib.rs file into subord_pkgid that has something in it
    writeFile(&package_dir.path().join_many([~"src", subord_pkgid.to_str(), ~"lib.rs"]),
              "pub fn f() {}");
    package_dir
}

fn create_local_package_with_custom_build_hook(pkgid: &PkgId,
                                               custom_build_hook: &str) -> TempDir {
    debug!("Dry run -- would create package {} with custom build hook {}",
           pkgid.to_str(), custom_build_hook);
    create_local_package(pkgid)
    // actually write the pkg.rs with the custom build hook

}

fn assert_lib_exists(repo: &Path, pkg_path: &Path, v: Version) {
    assert!(lib_exists(repo, pkg_path, v));
}

fn lib_exists(repo: &Path, pkg_path: &Path, _v: Version) -> bool { // ??? version?
    debug!("assert_lib_exists: repo = {}, pkg_path = {}", repo.display(), pkg_path.display());
    let lib = installed_library_in_workspace(pkg_path, repo);
    debug!("assert_lib_exists: checking whether {:?} exists", lib);
    lib.is_some() && {
        let libname = lib.get_ref();
        libname.exists() && is_rwx(libname)
    }
}

fn assert_executable_exists(repo: &Path, short_name: &str) {
    assert!(executable_exists(repo, short_name));
}

fn executable_exists(repo: &Path, short_name: &str) -> bool {
    debug!("executable_exists: repo = {}, short_name = {}", repo.display(), short_name);
    let exec = target_executable_in_workspace(&PkgId::new(short_name), repo);
    exec.exists() && is_rwx(&exec)
}

fn test_executable_exists(repo: &Path, short_name: &str) -> bool {
    debug!("test_executable_exists: repo = {}, short_name = {}", repo.display(), short_name);
    let exec = built_test_in_workspace(&PkgId::new(short_name), repo);
    exec.map_default(false, |exec| exec.exists() && is_rwx(&exec))
}

fn remove_executable_file(p: &PkgId, workspace: &Path) {
    let exec = target_executable_in_workspace(&PkgId::new(p.short_name), workspace);
    if exec.exists() {
        fs::unlink(&exec);
    }
}

fn assert_built_executable_exists(repo: &Path, short_name: &str) {
    assert!(built_executable_exists(repo, short_name));
}

fn built_executable_exists(repo: &Path, short_name: &str) -> bool {
    debug!("assert_built_executable_exists: repo = {}, short_name = {}",
            repo.display(), short_name);
    let exec = built_executable_in_workspace(&PkgId::new(short_name), repo);
    exec.is_some() && {
       let execname = exec.get_ref();
       execname.exists() && is_rwx(execname)
    }
}

fn remove_built_executable_file(p: &PkgId, workspace: &Path) {
    let exec = built_executable_in_workspace(&PkgId::new(p.short_name), workspace);
    match exec {
        Some(r) => fs::unlink(&r),
        None    => ()
    }
}

fn object_file_exists(repo: &Path, short_name: &str) -> bool {
    file_exists(repo, short_name, "o")
}

fn assembly_file_exists(repo: &Path, short_name: &str) -> bool {
    file_exists(repo, short_name, "s")
}

fn llvm_assembly_file_exists(repo: &Path, short_name: &str) -> bool {
    file_exists(repo, short_name, "ll")
}

fn llvm_bitcode_file_exists(repo: &Path, short_name: &str) -> bool {
    file_exists(repo, short_name, "bc")
}

fn file_exists(repo: &Path, short_name: &str, extension: &str) -> bool {
    target_build_dir(repo).join_many([short_name.to_owned(),
                                     format!("{}.{}", short_name, extension)])
                          .exists()
}

fn assert_built_library_exists(repo: &Path, short_name: &str) {
    assert!(built_library_exists(repo, short_name));
}

fn built_library_exists(repo: &Path, short_name: &str) -> bool {
    debug!("assert_built_library_exists: repo = {}, short_name = {}", repo.display(), short_name);
    let lib = built_library_in_workspace(&PkgId::new(short_name), repo);
    lib.is_some() && {
        let libname = lib.get_ref();
        libname.exists() && is_rwx(libname)
    }
}

fn command_line_test_output(args: &[~str]) -> ~[~str] {
    let mut result = ~[];
    let p_output = command_line_test(args, &os::getcwd());
    let test_output = str::from_utf8(p_output.output);
    for s in test_output.split('\n') {
        result.push(s.to_owned());
    }
    result
}

fn command_line_test_output_with_env(args: &[~str], env: ~[(~str, ~str)]) -> ~[~str] {
    let mut result = ~[];
    let p_output = match command_line_test_with_env(args,
        &os::getcwd(), Some(env)) {
        Fail(_) => fail!("Command-line test failed"),
        Success(r) => r
    };
    let test_output = str::from_utf8(p_output.output);
    for s in test_output.split('\n') {
        result.push(s.to_owned());
    }
    result
}

// assumes short_name and path are one and the same -- I should fix
fn lib_output_file_name(workspace: &Path, short_name: &str) -> Path {
    debug!("lib_output_file_name: given {} and short name {}",
           workspace.display(), short_name);
    library_in_workspace(&Path::new(short_name),
                         short_name,
                         Build,
                         workspace,
                         "build",
                         &NoVersion).expect("lib_output_file_name")
}

fn output_file_name(workspace: &Path, short_name: ~str) -> Path {
    target_build_dir(workspace).join(short_name.as_slice()).join(format!("{}{}", short_name,
                                                                         os::EXE_SUFFIX))
}

#[cfg(target_os = "linux")]
fn touch_source_file(workspace: &Path, pkgid: &PkgId) {
    use conditions::bad_path::cond;
    let pkg_src_dir = workspace.join_many([~"src", pkgid.to_str()]);
    let contents = fs::readdir(&pkg_src_dir);
    for p in contents.iter() {
        if p.extension_str() == Some("rs") {
            // should be able to do this w/o a process
            // FIXME (#9639): This needs to handle non-utf8 paths
            // n.b. Bumps time up by 2 seconds to get around granularity issues
            if !run::process_output("touch", [~"--date",
                                             ~"+2 seconds",
                                             p.as_str().unwrap().to_owned()])
                .expect("failed to exec `touch`").status.success() {
                let _ = cond.raise((pkg_src_dir.clone(), ~"Bad path"));
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn touch_source_file(workspace: &Path, pkgid: &PkgId) {
    use conditions::bad_path::cond;
    let pkg_src_dir = workspace.join_many([~"src", pkgid.to_str()]);
    let contents = fs::readdir(&pkg_src_dir);
    for p in contents.iter() {
        if p.extension_str() == Some("rs") {
            // should be able to do this w/o a process
            // FIXME (#9639): This needs to handle non-utf8 paths
            // n.b. Bumps time up by 2 seconds to get around granularity issues
            if !run::process_output("touch", [~"-A02",
                                             p.as_str().unwrap().to_owned()])
                .expect("failed to exec `touch`").status.success() {
                let _ = cond.raise((pkg_src_dir.clone(), ~"Bad path"));
            }
        }
    }
}

/// Add a comment at the end
fn frob_source_file(workspace: &Path, pkgid: &PkgId, filename: &str) {
    use conditions::bad_path::cond;
    let pkg_src_dir = workspace.join_many([~"src", pkgid.to_str()]);
    let mut maybe_p = None;
    let maybe_file = pkg_src_dir.join(filename);
    debug!("Trying to frob {} -- {}", pkg_src_dir.display(), filename);
    if maybe_file.exists() {
        maybe_p = Some(maybe_file);
    }
    debug!("Frobbed? {:?}", maybe_p);
    match maybe_p {
        Some(ref p) => {
            io::io_error::cond.trap(|e| {
                cond.raise((p.clone(), format!("Bad path: {}", e.desc)));
            }).inside(|| {
                let mut w = File::open_mode(p, io::Append, io::Write);
                w.write(bytes!("/* hi */\n"));
            })
        }
        None => fail!("frob_source_file failed to find a source file in {}",
                           pkg_src_dir.display())
    }
}

#[test]
fn test_make_dir_rwx() {
    let temp = &os::tmpdir();
    let dir = temp.join("quux");
    if dir.exists() {
        fs::rmdir_recursive(&dir);
    }
    debug!("Trying to make {}", dir.display());
    assert!(make_dir_rwx(&dir));
    assert!(dir.is_dir());
    assert!(is_rwx(&dir));
    fs::rmdir_recursive(&dir);
}

// n.b. I ignored the next two tests for now because something funny happens on linux
// and I don't want to debug the issue right now (calling into the rustpkg lib directly
// is a little sketchy anyway)
#[test]
#[ignore]
fn test_install_valid() {
    use path_util::installed_library_in_workspace;

    let sysroot = test_sysroot();
    debug!("sysroot = {}", sysroot.display());
    let temp_pkg_id = fake_pkg();
    let (temp_workspace, _pkg_dir) = mk_temp_workspace(&temp_pkg_id.path, &NoVersion);
    let temp_workspace = temp_workspace.path();
    let ctxt = fake_ctxt(sysroot, temp_workspace);
    debug!("temp_workspace = {}", temp_workspace.display());
    // should have test, bench, lib, and main
    let src = PkgSrc::new(temp_workspace.clone(),
                          temp_workspace.clone(),
                          false,
                          temp_pkg_id.clone());
    ctxt.install(src, &WhatToBuild::new(MaybeCustom, Everything));
    // Check that all files exist
    let exec = target_executable_in_workspace(&temp_pkg_id, temp_workspace);
    debug!("exec = {}", exec.display());
    assert!(exec.exists());
    assert!(is_rwx(&exec));

    let lib = installed_library_in_workspace(&temp_pkg_id.path, temp_workspace);
    debug!("lib = {:?}", lib);
    assert!(lib.as_ref().map_default(false, |l| l.exists()));
    assert!(lib.as_ref().map_default(false, |l| is_rwx(l)));

    // And that the test and bench executables aren't installed
    assert!(!target_test_in_workspace(&temp_pkg_id, temp_workspace).exists());
    let bench = target_bench_in_workspace(&temp_pkg_id, temp_workspace);
    debug!("bench = {}", bench.display());
    assert!(!bench.exists());

    // Make sure the db isn't dirty, so that it doesn't try to save()
    // asynchronously after the temporary directory that it wants to save
    // to has been deleted.
    ctxt.workcache_context.db.write(|db| db.db_dirty = false);
}

#[test]
#[ignore]
fn test_install_invalid() {
    let sysroot = test_sysroot();
    let pkgid = fake_pkg();
    let temp_workspace = TempDir::new("test").expect("couldn't create temp dir");
    let temp_workspace = temp_workspace.path().clone();
    let ctxt = fake_ctxt(sysroot, &temp_workspace);

    // Uses task::try because of #9001
    let result = do task::try {
        let pkg_src = PkgSrc::new(temp_workspace.clone(),
                                  temp_workspace.clone(),
                                  false,
                                  pkgid.clone());
        ctxt.install(pkg_src, &WhatToBuild::new(MaybeCustom, Everything));
    };
    assert!(result.unwrap_err()
            .to_str().contains("supplied path for package dir does not exist"));
}

#[test]
fn test_install_valid_external() {
    let temp_pkg_id = PkgId::new("foo");
    let (tempdir, _) = mk_temp_workspace(&temp_pkg_id.path,
                                         &temp_pkg_id.version);
    let temp_workspace = tempdir.path();
    command_line_test([~"install", ~"foo"], temp_workspace);

    // Check that all files exist
    let exec = target_executable_in_workspace(&temp_pkg_id, temp_workspace);
    debug!("exec = {}", exec.display());
    assert!(exec.exists());
    assert!(is_rwx(&exec));

    let lib = installed_library_in_workspace(&temp_pkg_id.path, temp_workspace);
    debug!("lib = {:?}", lib);
    assert!(lib.as_ref().map_default(false, |l| l.exists()));
    assert!(lib.as_ref().map_default(false, |l| is_rwx(l)));

    // And that the test and bench executables aren't installed
    assert!(!target_test_in_workspace(&temp_pkg_id, temp_workspace).exists());
    let bench = target_bench_in_workspace(&temp_pkg_id, temp_workspace);
    debug!("bench = {}", bench.display());
    assert!(!bench.exists());

}

#[test]
#[ignore(reason = "9994")]
fn test_install_invalid_external() {
    let cwd = os::getcwd();
    command_line_test_expect_fail([~"install", ~"foo"],
                                  &cwd,
                                  None,
                                  // FIXME #3408: Should be NONEXISTENT_PACKAGE_CODE
                                  COPY_FAILED_CODE);
}

#[test]
fn test_install_git() {
    let temp_pkg_id = git_repo_pkg();
    let repo = init_git_repo(&temp_pkg_id.path);
    let repo = repo.path();
    debug!("repo = {}", repo.display());
    let repo_subdir = repo.join_many(["mockgithub.com", "catamorphism", "test-pkg"]);
    debug!("repo_subdir = {}", repo_subdir.display());

    writeFile(&repo_subdir.join("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&repo_subdir.join("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&repo_subdir.join("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&repo_subdir.join("bench.rs"),
              "#[bench] pub fn f() { (); }");
    add_git_tag(&repo_subdir, ~"0.1"); // this has the effect of committing the files

    debug!("test_install_git: calling rustpkg install {} in {}",
           temp_pkg_id.path.display(), repo.display());
    // should have test, bench, lib, and main
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([~"install", temp_pkg_id.path.as_str().unwrap().to_owned()], repo);
    let ws = repo.join(".rust");
    // Check that all files exist
    debug!("Checking for files in {}", ws.display());
    let exec = target_executable_in_workspace(&temp_pkg_id, &ws);
    debug!("exec = {}", exec.display());
    assert!(exec.exists());
    assert!(is_rwx(&exec));
    let _built_lib =
        built_library_in_workspace(&temp_pkg_id,
                                   &ws).expect("test_install_git: built lib should exist");
    assert_lib_exists(&ws, &temp_pkg_id.path, temp_pkg_id.version.clone());
    let built_test = built_test_in_workspace(&temp_pkg_id,
                         &ws).expect("test_install_git: built test should exist");
    assert!(built_test.exists());
    let built_bench = built_bench_in_workspace(&temp_pkg_id,
                          &ws).expect("test_install_git: built bench should exist");
    assert!(built_bench.exists());
    // And that the test and bench executables aren't installed
    let test = target_test_in_workspace(&temp_pkg_id, &ws);
    assert!(!test.exists());
    debug!("test = {}", test.display());
    let bench = target_bench_in_workspace(&temp_pkg_id, &ws);
    debug!("bench = {}", bench.display());
    assert!(!bench.exists());
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

    assert_eq!(~"foo-0.0", whatever.to_str());
    assert!("github.com/catamorphism/test-pkg-0.0" ==
            PkgId::new("github.com/catamorphism/test-pkg").to_str());

    cond.trap(|(p, e)| {
        assert!(p.filename().is_none())
        assert!("0-length pkgid" == e);
        whatever.clone()
    }).inside(|| {
        let x = PkgId::new("");
        assert_eq!(~"foo-0.0", x.to_str());
    });

    cond.trap(|(p, e)| {
        let abs = os::make_absolute(&Path::new("foo/bar/quux"));
        assert_eq!(p, abs);
        assert!("absolute pkgid" == e);
        whatever.clone()
    }).inside(|| {
        let zp = os::make_absolute(&Path::new("foo/bar/quux"));
        // FIXME (#9639): This needs to handle non-utf8 paths
        let z = PkgId::new(zp.as_str().unwrap());
        assert_eq!(~"foo-0.0", z.to_str());
    })
}

#[test]
fn test_package_version() {
    let local_path = "mockgithub.com/catamorphism/test_pkg_version";
    let repo = init_git_repo(&Path::new(local_path));
    let repo = repo.path();
    let repo_subdir = repo.join_many(["mockgithub.com", "catamorphism", "test_pkg_version"]);
    debug!("Writing files in: {}", repo_subdir.display());
    fs::mkdir_recursive(&repo_subdir, io::UserRWX);
    writeFile(&repo_subdir.join("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&repo_subdir.join("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&repo_subdir.join("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&repo_subdir.join("bench.rs"),
              "#[bench] pub fn f() { (); }");
    add_git_tag(&repo_subdir, ~"0.4");

    // It won't pick up the 0.4 version because the dir isn't in the RUST_PATH, but...
    let temp_pkg_id = PkgId::new("mockgithub.com/catamorphism/test_pkg_version");
    // This should look at the prefix, clone into a workspace, then build.
    command_line_test([~"install", ~"mockgithub.com/catamorphism/test_pkg_version"],
                      repo);
    let ws = repo.join(".rust");
    // we can still match on the filename to make sure it contains the 0.4 version
    assert!(match built_library_in_workspace(&temp_pkg_id,
                                             &ws) {
        Some(p) => {
            let suffix = format!("0.4{}", os::consts::DLL_SUFFIX);
            p.as_vec().ends_with(suffix.as_bytes())
        }
        None    => false
    });
    assert!(built_executable_in_workspace(&temp_pkg_id, &ws)
            == Some(target_build_dir(&ws).join_many(["mockgithub.com",
                                                     "catamorphism",
                                                     "test_pkg_version",
                                                     "test_pkg_version"])));
}

#[test]
fn test_package_request_version() {
    let local_path = "mockgithub.com/catamorphism/test_pkg_version";
    let repo = init_git_repo(&Path::new(local_path));
    let repo = repo.path();
    let repo_subdir = repo.join_many(["mockgithub.com", "catamorphism", "test_pkg_version"]);
    debug!("Writing files in: {}", repo_subdir.display());
    writeFile(&repo_subdir.join("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&repo_subdir.join("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&repo_subdir.join("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&repo_subdir.join("bench.rs"),
              "#[bench] pub fn f() { (); }");
    writeFile(&repo_subdir.join("version-0.3-file.txt"), "hi");
    add_git_tag(&repo_subdir, ~"0.3");
    writeFile(&repo_subdir.join("version-0.4-file.txt"), "hello");
    add_git_tag(&repo_subdir, ~"0.4");

    command_line_test([~"install", format!("{}\\#0.3", local_path)], repo);

    assert!(match installed_library_in_workspace(&Path::new("test_pkg_version"),
                                                 &repo.join(".rust")) {
        Some(p) => {
            debug!("installed: {}", p.display());
            let suffix = format!("0.3{}", os::consts::DLL_SUFFIX);
            p.as_vec().ends_with(suffix.as_bytes())
        }
        None    => false
    });
    let temp_pkg_id = PkgId::new("mockgithub.com/catamorphism/test_pkg_version#0.3");
    assert!(target_executable_in_workspace(&temp_pkg_id, &repo.join(".rust"))
            == repo.join_many([".rust", "bin", "test_pkg_version"]));

    let mut dir = target_build_dir(&repo.join(".rust"));
    dir.push(&Path::new("src/mockgithub.com/catamorphism/test_pkg_version-0.3"));
    debug!("dir = {}", dir.display());
    assert!(dir.is_dir());
    assert!(dir.join("version-0.3-file.txt").exists());
    assert!(!dir.join("version-0.4-file.txt").exists());
}

#[test]
#[ignore (reason = "http-client not ported to rustpkg yet")]
fn rustpkg_install_url_2() {
    let temp_dir = TempDir::new("rustpkg_install_url_2").expect("rustpkg_install_url_2");
    command_line_test([~"install", ~"github.com/mozilla-servo/rust-http-client"],
                     temp_dir.path());
}

#[test]
fn rustpkg_library_target() {
    let foo_repo = init_git_repo(&Path::new("foo"));
    let foo_repo = foo_repo.path();
    let package_dir = foo_repo.join("foo");

    debug!("Writing files in: {}", package_dir.display());
    writeFile(&package_dir.join("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&package_dir.join("lib.rs"),
              "pub fn f() { let _x = (); }");
    writeFile(&package_dir.join("test.rs"),
              "#[test] pub fn f() { (); }");
    writeFile(&package_dir.join("bench.rs"),
              "#[bench] pub fn f() { (); }");

    add_git_tag(&package_dir, ~"1.0");
    command_line_test([~"install", ~"foo"], foo_repo);
    assert_lib_exists(&foo_repo.join(".rust"), &Path::new("foo"), ExactRevision(~"1.0"));
}

#[test]
fn rustpkg_local_pkg() {
    let dir = create_local_package(&PkgId::new("foo"));
    command_line_test([~"install", ~"foo"], dir.path());
    assert_executable_exists(dir.path(), "foo");
}

#[test]
#[ignore(reason="busted")]
fn package_script_with_default_build() {
    let dir = create_local_package(&PkgId::new("fancy-lib"));
    let dir = dir.path();
    debug!("dir = {}", dir.display());
    let mut source = test_sysroot().dir_path();
    source.pop(); source.pop();
    let source = Path::new(file!()).dir_path().join_many(
        [~"testsuite", ~"pass", ~"src", ~"fancy-lib", ~"pkg.rs"]);
    debug!("package_script_with_default_build: {}", source.display());
    fs::copy(&source, &dir.join_many(["src", "fancy-lib-0.0", "pkg.rs"]));
    command_line_test([~"install", ~"fancy-lib"], dir);
    assert_lib_exists(dir, &Path::new("fancy-lib"), NoVersion);
    assert!(target_build_dir(dir).join_many([~"fancy-lib", ~"generated.rs"]).exists());
    let generated_path = target_build_dir(dir).join_many([~"fancy-lib", ~"generated.rs"]);
    debug!("generated path = {}", generated_path.display());
    assert!(generated_path.exists());
}

#[test]
fn rustpkg_build_no_arg() {
    let tmp = TempDir::new("rustpkg_build_no_arg").expect("rustpkg_build_no_arg failed");
    let tmp = tmp.path().join(".rust");
    let package_dir = tmp.join_many(["src", "foo"]);
    fs::mkdir_recursive(&package_dir, io::UserRWX);

    writeFile(&package_dir.join("main.rs"),
              "fn main() { let _x = (); }");
    debug!("build_no_arg: dir = {}", package_dir.display());
    command_line_test([~"build"], &package_dir);
    assert_built_executable_exists(&tmp, "foo");
}

#[test]
fn rustpkg_install_no_arg() {
    let tmp = TempDir::new("rustpkg_install_no_arg").expect("rustpkg_install_no_arg failed");
    let tmp = tmp.path().join(".rust");
    let package_dir = tmp.join_many(["src", "foo"]);
    fs::mkdir_recursive(&package_dir, io::UserRWX);
    writeFile(&package_dir.join("lib.rs"),
              "fn main() { let _x = (); }");
    debug!("install_no_arg: dir = {}", package_dir.display());
    command_line_test([~"install"], &package_dir);
    assert_lib_exists(&tmp, &Path::new("foo"), NoVersion);
}

#[test]
fn rustpkg_clean_no_arg() {
    let tmp = TempDir::new("rustpkg_clean_no_arg").expect("rustpkg_clean_no_arg failed");
    let tmp = tmp.path().join(".rust");
    let package_dir = tmp.join_many(["src", "foo"]);
    fs::mkdir_recursive(&package_dir, io::UserRWX);

    writeFile(&package_dir.join("main.rs"),
              "fn main() { let _x = (); }");
    debug!("clean_no_arg: dir = {}", package_dir.display());
    command_line_test([~"build"], &package_dir);
    assert_built_executable_exists(&tmp, "foo");
    command_line_test([~"clean"], &package_dir);
    let res = built_executable_in_workspace(&PkgId::new("foo"), &tmp);
    assert!(!res.as_ref().map_default(false, |m| m.exists()));
}

#[test]
fn rust_path_test() {
    let dir_for_path = TempDir::new("more_rust").expect("rust_path_test failed");
    let dir = mk_workspace(dir_for_path.path(), &Path::new("foo"), &NoVersion);
    debug!("dir = {}", dir.display());
    writeFile(&dir.join("main.rs"), "fn main() { let _x = (); }");

    let cwd = os::getcwd();
    debug!("cwd = {}", cwd.display());
                                     // use command_line_test_with_env
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test_with_env([~"install", ~"foo"],
                               &cwd,
                               Some(~[(~"RUST_PATH",
                                       dir_for_path.path().as_str().unwrap().to_owned())]));
    assert_executable_exists(dir_for_path.path(), "foo");
}

#[test]
#[ignore] // FIXME(#9184) tests can't change the cwd (other tests are sad then)
fn rust_path_contents() {
    let dir = TempDir::new("rust_path").expect("rust_path_contents failed");
    let abc = &dir.path().join_many(["A", "B", "C"]);
    fs::mkdir_recursive(&abc.join(".rust"), io::UserRWX);
    fs::mkdir_recursive(&abc.with_filename(".rust"), io::UserRWX);
    fs::mkdir_recursive(&abc.dir_path().with_filename(".rust"), io::UserRWX);
    assert!(os::change_dir(abc));

    let p = rust_path();
    let cwd = os::getcwd().join(".rust");
    let parent = cwd.dir_path().with_filename(".rust");
    let grandparent = cwd.dir_path().dir_path().with_filename(".rust");
    assert!(p.contains(&cwd));
    assert!(p.contains(&parent));
    assert!(p.contains(&grandparent));
    for a_path in p.iter() {
        assert!(a_path.filename().is_some());
    }
}

#[test]
fn rust_path_parse() {
    os::setenv("RUST_PATH", "/a/b/c:/d/e/f:/g/h/i");
    let paths = rust_path();
    assert!(paths.contains(&Path::new("/g/h/i")));
    assert!(paths.contains(&Path::new("/d/e/f")));
    assert!(paths.contains(&Path::new("/a/b/c")));
    os::unsetenv("RUST_PATH");
}

#[test]
fn test_list() {
    let dir = TempDir::new("test_list").expect("test_list failed");
    let dir = dir.path();
    let foo = PkgId::new("foo");
    create_local_package_in(&foo, dir);
    let bar = PkgId::new("bar");
    create_local_package_in(&bar, dir);
    let quux = PkgId::new("quux");
    create_local_package_in(&quux, dir);

// list doesn't output very much right now...
    command_line_test([~"install", ~"foo"], dir);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let env_arg = ~[(~"RUST_PATH", dir.as_str().unwrap().to_owned())];
    let list_output = command_line_test_output_with_env([~"list"], env_arg.clone());
    assert!(list_output.iter().any(|x| x.starts_with("foo")));

    command_line_test([~"install", ~"bar"], dir);
    let list_output = command_line_test_output_with_env([~"list"], env_arg.clone());
    assert!(list_output.iter().any(|x| x.starts_with("foo")));
    assert!(list_output.iter().any(|x| x.starts_with("bar")));

    command_line_test([~"install", ~"quux"], dir);
    let list_output = command_line_test_output_with_env([~"list"], env_arg);
    assert!(list_output.iter().any(|x| x.starts_with("foo")));
    assert!(list_output.iter().any(|x| x.starts_with("bar")));
    assert!(list_output.iter().any(|x| x.starts_with("quux")));
}

#[test]
fn install_remove() {
    let dir = TempDir::new("install_remove").expect("install_remove");
    let dir = dir.path();
    let foo = PkgId::new("foo");
    let bar = PkgId::new("bar");
    let quux = PkgId::new("quux");
    create_local_package_in(&foo, dir);
    create_local_package_in(&bar, dir);
    create_local_package_in(&quux, dir);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let rust_path_to_use = ~[(~"RUST_PATH", dir.as_str().unwrap().to_owned())];
    command_line_test([~"install", ~"foo"], dir);
    command_line_test([~"install", ~"bar"], dir);
    command_line_test([~"install", ~"quux"], dir);
    let list_output = command_line_test_output_with_env([~"list"], rust_path_to_use.clone());
    assert!(list_output.iter().any(|x| x.starts_with("foo")));
    assert!(list_output.iter().any(|x| x.starts_with("bar")));
    assert!(list_output.iter().any(|x| x.starts_with("quux")));
    command_line_test([~"uninstall", ~"foo"], dir);
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
    let dir = TempDir::new("install_remove").expect("install_remove");
    let dir = dir.path();
    let foo = PkgId::new("foo");
    create_local_package_in(&foo, dir);

    command_line_test([~"install", ~"foo"], dir);
    command_line_test([~"install", ~"foo"], dir);
    let mut contents = ~[];
    let check_dups = |p: &PkgId| {
        if contents.contains(p) {
            fail!("package {} appears in `list` output more than once", p.path.display());
        }
        else {
            contents.push((*p).clone());
        }
        true
    };
    list_installed_packages(check_dups);
}

#[test]
fn no_rebuilding() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    command_line_test([~"build", ~"foo"], workspace);
    let foo_lib = lib_output_file_name(workspace, "foo");
    // Now make `foo` read-only so that subsequent rebuilds of it will fail
    assert!(chmod_read_only(&foo_lib));

    command_line_test([~"build", ~"foo"], workspace);

    match command_line_test_partial([~"build", ~"foo"], workspace) {
        Success(..) => (), // ok
        Fail(ref status) if status.status.matches_exit_status(65) =>
            fail!("no_rebuilding failed: it tried to rebuild bar"),
        Fail(_) => fail!("no_rebuilding failed for some other reason")
    }
}

#[test]
#[ignore]
fn no_recopying() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    command_line_test([~"install", ~"foo"], workspace);
    let foo_lib = installed_library_in_workspace(&p_id.path, workspace);
    assert!(foo_lib.is_some());
    // Now make `foo` read-only so that subsequent attempts to copy to it will fail
    assert!(chmod_read_only(&foo_lib.unwrap()));

    match command_line_test_partial([~"install", ~"foo"], workspace) {
        Success(..) => (), // ok
        Fail(ref status) if status.status.matches_exit_status(65) =>
            fail!("no_recopying failed: it tried to re-copy foo"),
        Fail(_) => fail!("no_copying failed for some other reason")
    }
}

#[test]
fn no_rebuilding_dep() {
    let p_id = PkgId::new("foo");
    let dep_id = PkgId::new("bar");
    let workspace = create_local_package_with_dep(&p_id, &dep_id);
    let workspace = workspace.path();
    command_line_test([~"build", ~"foo"], workspace);
    let bar_lib = lib_output_file_name(workspace, "bar");
    frob_source_file(workspace, &p_id, "main.rs");
    // Now make `bar` read-only so that subsequent rebuilds of it will fail
    assert!(chmod_read_only(&bar_lib));
    match command_line_test_partial([~"build", ~"foo"], workspace) {
        Success(..) => (), // ok
        Fail(ref r) if r.status.matches_exit_status(65) =>
            fail!("no_rebuilding_dep failed: it tried to rebuild bar"),
        Fail(_) => fail!("no_rebuilding_dep failed for some other reason")
    }
}

#[test]
fn do_rebuild_dep_dates_change() {
    let p_id = PkgId::new("foo");
    let dep_id = PkgId::new("bar");
    let workspace = create_local_package_with_dep(&p_id, &dep_id);
    let workspace = workspace.path();
    command_line_test([~"build", ~"foo"], workspace);
    let bar_lib_name = lib_output_file_name(workspace, "bar");
    touch_source_file(workspace, &dep_id);

    // Now make `bar` read-only so that subsequent rebuilds of it will fail
    assert!(chmod_read_only(&bar_lib_name));

    match command_line_test_partial([~"build", ~"foo"], workspace) {
        Success(..) => fail!("do_rebuild_dep_dates_change failed: it didn't rebuild bar"),
        Fail(ref r) if r.status.matches_exit_status(65) => (), // ok
        Fail(_) => fail!("do_rebuild_dep_dates_change failed for some other reason")
    }
}

#[test]
fn do_rebuild_dep_only_contents_change() {
    let p_id = PkgId::new("foo");
    let dep_id = PkgId::new("bar");
    let workspace = create_local_package_with_dep(&p_id, &dep_id);
    let workspace = workspace.path();
    command_line_test([~"build", ~"foo"], workspace);
    frob_source_file(workspace, &dep_id, "lib.rs");
    let bar_lib_name = lib_output_file_name(workspace, "bar");

    // Now make `bar` read-only so that subsequent rebuilds of it will fail
    assert!(chmod_read_only(&bar_lib_name));

    // should adjust the datestamp
    match command_line_test_partial([~"build", ~"foo"], workspace) {
        Success(..) => fail!("do_rebuild_dep_only_contents_change failed: it didn't rebuild bar"),
        Fail(ref r) if r.status.matches_exit_status(65) => (), // ok
        Fail(_) => fail!("do_rebuild_dep_only_contents_change failed for some other reason")
    }
}

#[test]
fn test_versions() {
    let workspace = create_local_package(&PkgId::new("foo#0.1"));
    let _other_workspace = create_local_package(&PkgId::new("foo#0.2"));
    command_line_test([~"install", ~"foo#0.1"], workspace.path());
    let output = command_line_test_output([~"list"]);
    // make sure output includes versions
    assert!(!output.iter().any(|x| x == &~"foo#0.2"));
}

#[test]
#[ignore(reason = "do not yet implemented")]
fn test_build_hooks() {
    let workspace = create_local_package_with_custom_build_hook(&PkgId::new("foo"),
                                                                "frob");
    command_line_test([~"do", ~"foo", ~"frob"], workspace.path());
}


#[test]
#[ignore(reason = "info not yet implemented")]
fn test_info() {
    let expected_info = ~"package foo"; // fill in
    let workspace = create_local_package(&PkgId::new("foo"));
    let output = command_line_test([~"info", ~"foo"], workspace.path());
    assert_eq!(str::from_utf8_owned(output.output), expected_info);
}

#[test]
fn test_uninstall() {
    let workspace = create_local_package(&PkgId::new("foo"));
    command_line_test([~"uninstall", ~"foo"], workspace.path());
    let output = command_line_test([~"list"], workspace.path());
    assert!(!str::from_utf8(output.output).contains("foo"));
}

#[test]
fn test_non_numeric_tag() {
    let temp_pkg_id = git_repo_pkg();
    let repo = init_git_repo(&temp_pkg_id.path);
    let repo = repo.path();
    let repo_subdir = repo.join_many(["mockgithub.com", "catamorphism", "test-pkg"]);
    writeFile(&repo_subdir.join("foo"), "foo");
    writeFile(&repo_subdir.join("lib.rs"),
              "pub fn f() { let _x = (); }");
    add_git_tag(&repo_subdir, ~"testbranch");
    writeFile(&repo_subdir.join("testbranch_only"), "hello");
    add_git_tag(&repo_subdir, ~"another_tag");
    writeFile(&repo_subdir.join("not_on_testbranch_only"), "bye bye");
    add_all_and_commit(&repo_subdir);

    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([~"install", format!("{}\\#testbranch",
                                           temp_pkg_id.path.as_str().unwrap())], repo);
    let file1 = repo.join_many(["mockgithub.com", "catamorphism", "test-pkg", "testbranch_only"]);
    let file2 = repo.join_many(["mockgithub.com", "catamorphism", "test-pkg", "master_only"]);
    assert!(file1.exists());
    assert!(!file2.exists());
}

#[test]
fn test_extern_mod() {
    let dir = TempDir::new("test_extern_mod").expect("test_extern_mod");
    let dir = dir.path();
    let main_file = dir.join("main.rs");
    let lib_depend_dir = TempDir::new("foo").expect("test_extern_mod");
    let lib_depend_dir = lib_depend_dir.path();
    let aux_dir = lib_depend_dir.join_many(["src", "mockgithub.com", "catamorphism", "test_pkg"]);
    fs::mkdir_recursive(&aux_dir, io::UserRWX);
    let aux_pkg_file = aux_dir.join("lib.rs");

    writeFile(&aux_pkg_file, "pub mod bar { pub fn assert_true() {  assert!(true); } }\n");
    assert!(aux_pkg_file.exists());

    writeFile(&main_file,
              "extern mod test = \"mockgithub.com/catamorphism/test_pkg\";\nuse test::bar;\
               fn main() { bar::assert_true(); }\n");

    command_line_test([~"install", ~"mockgithub.com/catamorphism/test_pkg"], lib_depend_dir);

    let exec_file = dir.join("out");
    // Be sure to extend the existing environment
    // FIXME (#9639): This needs to handle non-utf8 paths
    let env = Some([(~"RUST_PATH", lib_depend_dir.as_str().unwrap().to_owned())] + os::env());
    let rustpkg_exec = rustpkg_exec();
    let rustc = rustpkg_exec.with_filename("rustc");

    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    let mut prog = run::Process::new(rustc.as_str().unwrap(),
                                     [main_file.as_str().unwrap().to_owned(),
                                      ~"--sysroot", test_sys.as_str().unwrap().to_owned(),
                                      ~"-o", exec_file.as_str().unwrap().to_owned()],
                                     run::ProcessOptions {
        env: env,
        dir: Some(dir),
        in_fd: None,
        out_fd: None,
        err_fd: None
    }).expect(format!("failed to exec `{}`", rustc.as_str().unwrap()));
    let outp = prog.finish_with_output();
    if !outp.status.success() {
        fail!("output was {}, error was {}",
              str::from_utf8(outp.output),
              str::from_utf8(outp.error));
    }
    assert!(exec_file.exists() && is_executable(&exec_file));
}

#[test]
fn test_extern_mod_simpler() {
    let dir = TempDir::new("test_extern_mod_simpler").expect("test_extern_mod_simpler");
    let dir = dir.path();
    let main_file = dir.join("main.rs");
    let lib_depend_dir = TempDir::new("foo").expect("test_extern_mod_simpler");
    let lib_depend_dir = lib_depend_dir.path();
    let aux_dir = lib_depend_dir.join_many(["src", "rust-awesomeness"]);
    fs::mkdir_recursive(&aux_dir, io::UserRWX);
    let aux_pkg_file = aux_dir.join("lib.rs");

    writeFile(&aux_pkg_file, "pub mod bar { pub fn assert_true() {  assert!(true); } }\n");
    assert!(aux_pkg_file.exists());

    writeFile(&main_file,
              "extern mod test = \"rust-awesomeness\";\nuse test::bar;\
               fn main() { bar::assert_true(); }\n");

    command_line_test([~"install", ~"rust-awesomeness"], lib_depend_dir);

    let exec_file = dir.join("out");
    // Be sure to extend the existing environment
    // FIXME (#9639): This needs to handle non-utf8 paths
    let env = Some([(~"RUST_PATH", lib_depend_dir.as_str().unwrap().to_owned())] + os::env());
    let rustpkg_exec = rustpkg_exec();
    let rustc = rustpkg_exec.with_filename("rustc");
    let test_sys = test_sysroot();
    debug!("RUST_PATH={} {} {} \n --sysroot {} -o {}",
                     lib_depend_dir.display(),
                     rustc.display(),
                     main_file.display(),
                     test_sys.display(),
                     exec_file.display());

    // FIXME (#9639): This needs to handle non-utf8 paths
    let mut prog = run::Process::new(rustc.as_str().unwrap(),
                                     [main_file.as_str().unwrap().to_owned(),
                                      ~"--sysroot", test_sys.as_str().unwrap().to_owned(),
                                      ~"-o", exec_file.as_str().unwrap().to_owned()],
                                     run::ProcessOptions {
        env: env,
        dir: Some(dir),
        in_fd: None,
        out_fd: None,
        err_fd: None
    }).expect(format!("failed to exec `{}`", rustc.as_str().unwrap()));
    let outp = prog.finish_with_output();
    if !outp.status.success() {
        fail!("output was {}, error was {}",
              str::from_utf8(outp.output),
              str::from_utf8(outp.error));
    }
    assert!(exec_file.exists() && is_executable(&exec_file));
}

#[test]
fn test_import_rustpkg() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    writeFile(&workspace.join_many(["src", "foo-0.0", "pkg.rs"]),
              "extern mod rustpkg; fn main() {}");
    command_line_test([~"build", ~"foo"], workspace);
    debug!("workspace = {}", workspace.display());
    assert!(target_build_dir(workspace).join("foo").join(format!("pkg{}",
        os::EXE_SUFFIX)).exists());
}

#[test]
fn test_macro_pkg_script() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    writeFile(&workspace.join_many(["src", "foo-0.0", "pkg.rs"]),
              "extern mod rustpkg; fn main() { debug!(\"Hi\"); }");
    command_line_test([~"build", ~"foo"], workspace);
    debug!("workspace = {}", workspace.display());
    assert!(target_build_dir(workspace).join("foo").join(format!("pkg{}",
        os::EXE_SUFFIX)).exists());
}

#[test]
fn multiple_workspaces() {
// Make a package foo; build/install in directory A
// Copy the exact same package into directory B and install it
// Set the RUST_PATH to A:B
// Make a third package that uses foo, make sure we can build/install it
    let (a_loc, _pkg_dir) = mk_temp_workspace(&Path::new("foo"), &NoVersion);
    let (b_loc, _pkg_dir) = mk_temp_workspace(&Path::new("foo"), &NoVersion);
    let (a_loc, b_loc) = (a_loc.path(), b_loc.path());
    debug!("Trying to install foo in {}", a_loc.display());
    command_line_test([~"install", ~"foo"], a_loc);
    debug!("Trying to install foo in {}", b_loc.display());
    command_line_test([~"install", ~"foo"], b_loc);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let env = Some(~[(~"RUST_PATH", format!("{}:{}", a_loc.as_str().unwrap(),
                                            b_loc.as_str().unwrap()))]);
    let c_loc = create_local_package_with_dep(&PkgId::new("bar"), &PkgId::new("foo"));
    command_line_test_with_env([~"install", ~"bar"], c_loc.path(), env);
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
   let workspace = workspace.path();
   let dest_workspace = mk_empty_workspace(&Path::new("bar"), &NoVersion, "dest_workspace");
   let dest_workspace = dest_workspace.path();
   let foo_path = workspace.join_many(["src", "foo-0.0"]);
   let rust_path = Some(~[(~"RUST_PATH",
       format!("{}:{}",
               dest_workspace.as_str().unwrap(),
               foo_path.as_str().unwrap()))]);
   command_line_test_with_env(~[~"install"] + if hack_flag { ~[~"--rust-path-hack"] } else { ~[] } +
                               ~[~"foo"], dest_workspace, rust_path);
   assert_lib_exists(dest_workspace, &Path::new("foo"), NoVersion);
   assert_executable_exists(dest_workspace, "foo");
   assert_built_library_exists(dest_workspace, "foo");
   assert_built_executable_exists(dest_workspace, "foo");
   assert!(!lib_exists(workspace, &Path::new("foo"), NoVersion));
   assert!(!executable_exists(workspace, "foo"));
   assert!(!built_library_exists(workspace, "foo"));
   assert!(!built_executable_exists(workspace, "foo"));
}

// Notice that this is the only test case where the --rust-path-hack
// flag is actually needed
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
   let cwd = TempDir::new("foo").expect("rust_path_hack_cwd");
   let cwd = cwd.path().join("foo");
   fs::mkdir_recursive(&cwd, io::UserRWX);
   writeFile(&cwd.join("lib.rs"), "pub fn f() { }");

   let dest_workspace = mk_empty_workspace(&Path::new("bar"), &NoVersion, "dest_workspace");
   let dest_workspace = dest_workspace.path();
   // FIXME (#9639): This needs to handle non-utf8 paths
   let rust_path = Some(~[(~"RUST_PATH", dest_workspace.as_str().unwrap().to_owned())]);
   command_line_test_with_env([~"install", ~"--rust-path-hack", ~"foo"], &cwd, rust_path);
   debug!("Checking that foo exists in {}", dest_workspace.display());
   assert_lib_exists(dest_workspace, &Path::new("foo"), NoVersion);
   assert_built_library_exists(dest_workspace, "foo");
   assert!(!lib_exists(&cwd, &Path::new("foo"), NoVersion));
   assert!(!built_library_exists(&cwd, "foo"));
}

#[test]
fn rust_path_hack_multi_path() {
   // Same as rust_path_hack_test, but with a more complex package ID
   let cwd = TempDir::new("pkg_files").expect("rust_path_hack_cwd");
   let subdir = cwd.path().join_many(["foo", "bar", "quux"]);
   fs::mkdir_recursive(&subdir, io::UserRWX);
   writeFile(&subdir.join("lib.rs"), "pub fn f() { }");
   let name = ~"foo/bar/quux";

   let dest_workspace = mk_empty_workspace(&Path::new("bar"), &NoVersion, "dest_workspace");
   let dest_workspace = dest_workspace.path();
   // FIXME (#9639): This needs to handle non-utf8 paths
   let rust_path = Some(~[(~"RUST_PATH", dest_workspace.as_str().unwrap().to_owned())]);
   command_line_test_with_env([~"install", ~"--rust-path-hack", name.clone()], &subdir, rust_path);
   debug!("Checking that {} exists in {}", name, dest_workspace.display());
   assert_lib_exists(dest_workspace, &Path::new("quux"), NoVersion);
   assert_built_library_exists(dest_workspace, name);
   assert!(!lib_exists(&subdir, &Path::new("quux"), NoVersion));
   assert!(!built_library_exists(&subdir, name));
}

#[test]
fn rust_path_hack_install_no_arg() {
   // Same as rust_path_hack_cwd, but making rustpkg infer the pkg id
   let cwd = TempDir::new("pkg_files").expect("rust_path_hack_install_no_arg");
   let cwd = cwd.path();
   let source_dir = cwd.join("foo");
   assert!(make_dir_rwx(&source_dir));
   writeFile(&source_dir.join("lib.rs"), "pub fn f() { }");

   let dest_workspace = mk_empty_workspace(&Path::new("bar"), &NoVersion, "dest_workspace");
   let dest_workspace = dest_workspace.path();
   // FIXME (#9639): This needs to handle non-utf8 paths
   let rust_path = Some(~[(~"RUST_PATH", dest_workspace.as_str().unwrap().to_owned())]);
   command_line_test_with_env([~"install", ~"--rust-path-hack"], &source_dir, rust_path);
   debug!("Checking that foo exists in {}", dest_workspace.display());
   assert_lib_exists(dest_workspace, &Path::new("foo"), NoVersion);
   assert_built_library_exists(dest_workspace, "foo");
   assert!(!lib_exists(&source_dir, &Path::new("foo"), NoVersion));
   assert!(!built_library_exists(cwd, "foo"));
}

#[test]
fn rust_path_hack_build_no_arg() {
   // Same as rust_path_hack_install_no_arg, but building instead of installing
   let cwd = TempDir::new("pkg_files").expect("rust_path_hack_build_no_arg");
   let source_dir = cwd.path().join("foo");
   assert!(make_dir_rwx(&source_dir));
   writeFile(&source_dir.join("lib.rs"), "pub fn f() { }");

   let dest_workspace = mk_empty_workspace(&Path::new("bar"), &NoVersion, "dest_workspace");
   let dest_workspace = dest_workspace.path();
   // FIXME (#9639): This needs to handle non-utf8 paths
   let rust_path = Some(~[(~"RUST_PATH", dest_workspace.as_str().unwrap().to_owned())]);
   command_line_test_with_env([~"build", ~"--rust-path-hack"], &source_dir, rust_path);
   debug!("Checking that foo exists in {}", dest_workspace.display());
   assert_built_library_exists(dest_workspace, "foo");
   assert!(!built_library_exists(&source_dir, "foo"));
}

#[test]
fn rust_path_hack_build_with_dependency() {
    let foo_id = PkgId::new("foo");
    let dep_id = PkgId::new("dep");
    // Tests that when --rust-path-hack is in effect, dependencies get built
    // into the destination workspace and not the source directory
    let work_dir = create_local_package(&foo_id);
    let work_dir = work_dir.path();
    let dep_workspace = create_local_package(&dep_id);
    let dep_workspace = dep_workspace.path();
    let dest_workspace = mk_emptier_workspace("dep");
    let dest_workspace = dest_workspace.path();
    let source_dir = work_dir.join_many(["src", "foo-0.0"]);
    writeFile(&source_dir.join("lib.rs"), "extern mod dep; pub fn f() { }");
    let dep_dir = dep_workspace.join_many(["src", "dep-0.0"]);
    let rust_path = Some(~[(~"RUST_PATH",
                          format!("{}:{}",
                                  dest_workspace.display(),
                                  dep_dir.display()))]);
    command_line_test_with_env([~"build", ~"--rust-path-hack", ~"foo"], work_dir, rust_path);
    assert_built_library_exists(dest_workspace, "dep");
    assert!(!built_library_exists(dep_workspace, "dep"));
}

#[test]
fn rust_path_install_target() {
    let dir_for_path = TempDir::new(
        "source_workspace").expect("rust_path_install_target failed");
    let mut dir = mk_workspace(dir_for_path.path(), &Path::new("foo"), &NoVersion);
    debug!("dir = {}", dir.display());
    writeFile(&dir.join("main.rs"), "fn main() { let _x = (); }");
    let dir_to_install_to = TempDir::new(
        "dest_workspace").expect("rust_path_install_target failed");
    let dir_to_install_to = dir_to_install_to.path();
    dir.pop(); dir.pop();

    // FIXME (#9639): This needs to handle non-utf8 paths
    let rust_path = Some(~[(~"RUST_PATH", format!("{}:{}",
                                                  dir_to_install_to.as_str().unwrap(),
                                                  dir.as_str().unwrap()))]);
    let cwd = os::getcwd();
    command_line_test_with_env([~"install", ~"foo"],
                               &cwd,
                               rust_path);

    assert_executable_exists(dir_to_install_to, "foo");

}

#[test]
fn sysroot_flag() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    // no-op sysroot setting; I'm not sure how else to test this
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([~"--sysroot",
                       test_sys.as_str().unwrap().to_owned(),
                       ~"build",
                       ~"foo"],
                      workspace);
    assert_built_executable_exists(workspace, "foo");
}

#[test]
fn compile_flag_build() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([test_sys.as_str().unwrap().to_owned(),
                       ~"build",
                       ~"--no-link",
                       ~"foo"],
                      workspace);
    assert!(!built_executable_exists(workspace, "foo"));
    assert!(object_file_exists(workspace, "foo"));
}

#[test]
fn compile_flag_fail() {
    // --no-link shouldn't be accepted for install
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test_expect_fail([test_sys.as_str().unwrap().to_owned(),
                       ~"install",
                       ~"--no-link",
                       ~"foo"],
                      workspace, None, BAD_FLAG_CODE);
    assert!(!built_executable_exists(workspace, "foo"));
    assert!(!object_file_exists(workspace, "foo"));
}

#[test]
fn notrans_flag_build() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let flags_to_test = [~"--no-trans", ~"--parse-only",
                         ~"--pretty", ~"-S"];

    for flag in flags_to_test.iter() {
        let test_sys = test_sysroot();
        // FIXME (#9639): This needs to handle non-utf8 paths
        command_line_test([test_sys.as_str().unwrap().to_owned(),
                           ~"build",
                           flag.clone(),
                           ~"foo"],
                          workspace);
        // Ideally we'd test that rustpkg actually succeeds, but
        // since task failure doesn't set the exit code properly,
        // we can't tell
        assert!(!built_executable_exists(workspace, "foo"));
        assert!(!object_file_exists(workspace, "foo"));
    }
}

#[test]
fn notrans_flag_fail() {
    // --no-trans shouldn't be accepted for install
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let flags_to_test = [~"--no-trans", ~"--parse-only",
                         ~"--pretty", ~"-S"];
    for flag in flags_to_test.iter() {
        let test_sys = test_sysroot();
        // FIXME (#9639): This needs to handle non-utf8 paths
        command_line_test_expect_fail([test_sys.as_str().unwrap().to_owned(),
                           ~"install",
                           flag.clone(),
                           ~"foo"],
                          workspace, None, BAD_FLAG_CODE);
        assert!(!built_executable_exists(workspace, "foo"));
        assert!(!object_file_exists(workspace, "foo"));
        assert!(!lib_exists(workspace, &Path::new("foo"), NoVersion));
    }
}

#[test]
fn dash_S() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([test_sys.as_str().unwrap().to_owned(),
                       ~"build",
                       ~"-S",
                       ~"foo"],
                      workspace);
    assert!(!built_executable_exists(workspace, "foo"));
    assert!(!object_file_exists(workspace, "foo"));
    assert!(assembly_file_exists(workspace, "foo"));
}

#[test]
fn dash_S_fail() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test_expect_fail([test_sys.as_str().unwrap().to_owned(),
                       ~"install",
                       ~"-S",
                       ~"foo"],
                       workspace, None, BAD_FLAG_CODE);
    assert!(!built_executable_exists(workspace, "foo"));
    assert!(!object_file_exists(workspace, "foo"));
    assert!(!assembly_file_exists(workspace, "foo"));
}

#[test]
fn test_cfg_build() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    // If the cfg flag gets messed up, this won't compile
    writeFile(&workspace.join_many(["src", "foo-0.0", "main.rs"]),
               "#[cfg(quux)] fn main() {}");
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([test_sys.as_str().unwrap().to_owned(),
                       ~"build",
                       ~"--cfg",
                       ~"quux",
                       ~"foo"],
                      workspace);
    assert_built_executable_exists(workspace, "foo");
}

#[test]
fn test_cfg_fail() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    writeFile(&workspace.join_many(["src", "foo-0.0", "main.rs"]),
               "#[cfg(quux)] fn main() {}");
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    match command_line_test_partial([test_sys.as_str().unwrap().to_owned(),
                       ~"build",
                       ~"foo"],
                      workspace) {
        Success(..) => fail!("test_cfg_fail failed"),
        _          => ()
    }
}


#[test]
fn test_emit_llvm_S_build() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([test_sys.as_str().unwrap().to_owned(),
                       ~"build",
                       ~"-S", ~"--emit-llvm",
                       ~"foo"],
                      workspace);
    assert!(!built_executable_exists(workspace, "foo"));
    assert!(!object_file_exists(workspace, "foo"));
    assert!(llvm_assembly_file_exists(workspace, "foo"));
    assert!(!assembly_file_exists(workspace, "foo"));
}

#[test]
fn test_emit_llvm_S_fail() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test_expect_fail([test_sys.as_str().unwrap().to_owned(),
                       ~"install",
                       ~"-S", ~"--emit-llvm",
                       ~"foo"],
                       workspace,
                       None,
                       BAD_FLAG_CODE);
    assert!(!built_executable_exists(workspace, "foo"));
    assert!(!object_file_exists(workspace, "foo"));
    assert!(!llvm_assembly_file_exists(workspace, "foo"));
    assert!(!assembly_file_exists(workspace, "foo"));
}

#[test]
fn test_emit_llvm_build() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([test_sys.as_str().unwrap().to_owned(),
                       ~"build",
                       ~"--emit-llvm",
                       ~"foo"],
                      workspace);
    assert!(!built_executable_exists(workspace, "foo"));
    assert!(!object_file_exists(workspace, "foo"));
    assert!(llvm_bitcode_file_exists(workspace, "foo"));
    assert!(!assembly_file_exists(workspace, "foo"));
    assert!(!llvm_assembly_file_exists(workspace, "foo"));
}

#[test]
fn test_emit_llvm_fail() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test_expect_fail([test_sys.as_str().unwrap().to_owned(),
                       ~"install",
                       ~"--emit-llvm",
                       ~"foo"],
                                  workspace,
                                  None,
                                  BAD_FLAG_CODE);
    assert!(!built_executable_exists(workspace, "foo"));
    assert!(!object_file_exists(workspace, "foo"));
    assert!(!llvm_bitcode_file_exists(workspace, "foo"));
    assert!(!llvm_assembly_file_exists(workspace, "foo"));
    assert!(!assembly_file_exists(workspace, "foo"));
}

#[test]
fn test_linker_build() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let matches = getopts([], optgroups());
    let options = build_session_options(@"rustpkg",
                                        matches.as_ref().unwrap(),
                                        @diagnostic::DefaultEmitter as
                                            @diagnostic::Emitter);
    let sess = build_session(options,
                             @diagnostic::DefaultEmitter as
                                @diagnostic::Emitter);
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    let cc = get_cc_prog(sess);
    command_line_test([test_sys.as_str().unwrap().to_owned(),
                       ~"install",
                       ~"--linker",
                       cc,
                       ~"foo"],
                      workspace);
    assert_executable_exists(workspace, "foo");
}

#[test]
fn test_build_install_flags_fail() {
    // The following flags can only be used with build or install:
    let forbidden = [~[~"--linker", ~"ld"],
                     ~[~"--link-args", ~"quux"],
                     ~[~"-O"],
                     ~[~"--opt-level", ~"2"],
                     ~[~"--save-temps"],
                     ~[~"--target", host_triple()],
                     ~[~"--target-cpu", ~"generic"],
                     ~[~"-Z", ~"--time-passes"]];
    let cwd = os::getcwd();
    for flag in forbidden.iter() {
        let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
        command_line_test_expect_fail([test_sys.as_str().unwrap().to_owned(),
                           ~"list"] + *flag, &cwd, None, BAD_FLAG_CODE);
    }
}

#[test]
fn test_optimized_build() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([test_sys.as_str().unwrap().to_owned(),
                       ~"build",
                       ~"-O",
                       ~"foo"],
                      workspace);
    assert!(built_executable_exists(workspace, "foo"));
}

#[test]
fn pkgid_pointing_to_subdir() {
    // The actual repo is mockgithub.com/mozilla/some_repo
    // rustpkg should recognize that and treat the part after some_repo/ as a subdir
    let workspace = TempDir::new("parent_repo").expect("Couldn't create temp dir");
    let workspace = workspace.path();
    fs::mkdir_recursive(&workspace.join_many(["src", "mockgithub.com",
                                                "mozilla", "some_repo"]),
                          io::UserRWX);

    let foo_dir = workspace.join_many(["src", "mockgithub.com", "mozilla", "some_repo",
                                       "extras", "foo"]);
    let bar_dir = workspace.join_many(["src", "mockgithub.com", "mozilla", "some_repo",
                                       "extras", "bar"]);
    fs::mkdir_recursive(&foo_dir, io::UserRWX);
    fs::mkdir_recursive(&bar_dir, io::UserRWX);
    writeFile(&foo_dir.join("lib.rs"),
              "#[pkgid=\"mockgithub.com/mozilla/some_repo/extras/foo\"]; pub fn f() {}");
    writeFile(&bar_dir.join("lib.rs"),
              "#[pkgid=\"mockgithub.com/mozilla/some_repo/extras/bar\"]; pub fn g() {}");

    debug!("Creating a file in {}", workspace.display());
    let testpkg_dir = workspace.join_many(["src", "testpkg-0.0"]);
    fs::mkdir_recursive(&testpkg_dir, io::UserRWX);

    writeFile(&testpkg_dir.join("main.rs"),
              "extern mod foo = \"mockgithub.com/mozilla/some_repo/extras/foo\";\n
               extern mod bar = \"mockgithub.com/mozilla/some_repo/extras/bar\";\n
               use foo::f; use bar::g; \n
               fn main() { f(); g(); }");

    command_line_test([~"install", ~"testpkg"], workspace);
    assert_executable_exists(workspace, "testpkg");
}

#[test]
fn test_recursive_deps() {
    let a_id = PkgId::new("a");
    let b_id = PkgId::new("b");
    let c_id = PkgId::new("c");
    let b_workspace = create_local_package_with_dep(&b_id, &c_id);
    let b_workspace = b_workspace.path();
    writeFile(&b_workspace.join_many(["src", "c-0.0", "lib.rs"]),
               "pub fn g() {}");
    let a_workspace = create_local_package(&a_id);
    let a_workspace = a_workspace.path();
    writeFile(&a_workspace.join_many(["src", "a-0.0", "main.rs"]),
               "extern mod b; use b::f; fn main() { f(); }");
    writeFile(&b_workspace.join_many(["src", "b-0.0", "lib.rs"]),
               "extern mod c; use c::g; pub fn f() { g(); }");
    // FIXME (#9639): This needs to handle non-utf8 paths
    let environment = Some(~[(~"RUST_PATH", b_workspace.as_str().unwrap().to_owned())]);
    debug!("RUST_PATH={}", b_workspace.display());
    command_line_test_with_env([~"install", ~"a"],
                               a_workspace,
                               environment);
    assert_lib_exists(a_workspace, &Path::new("a"), NoVersion);
    assert_lib_exists(b_workspace, &Path::new("b"), NoVersion);
    assert_lib_exists(b_workspace, &Path::new("c"), NoVersion);
}

#[test]
fn test_install_to_rust_path() {
    let p_id = PkgId::new("foo");
    let second_workspace = create_local_package(&p_id);
    let second_workspace = second_workspace.path();
    let first_workspace = mk_empty_workspace(&Path::new("p"), &NoVersion, "dest");
    let first_workspace = first_workspace.path();
    // FIXME (#9639): This needs to handle non-utf8 paths
    let rust_path = Some(~[(~"RUST_PATH",
                            format!("{}:{}", first_workspace.as_str().unwrap(),
                                    second_workspace.as_str().unwrap()))]);
    debug!("RUST_PATH={}:{}", first_workspace.display(), second_workspace.display());
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test_with_env([test_sys.as_str().unwrap().to_owned(),
                       ~"install",
                       ~"foo"],
                      &os::getcwd(), rust_path);
    assert!(!built_executable_exists(first_workspace, "foo"));
    assert!(built_executable_exists(second_workspace, "foo"));
    assert_executable_exists(first_workspace, "foo");
    assert!(!executable_exists(second_workspace, "foo"));
}

#[test]
fn test_target_specific_build_dir() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([test_sys.as_str().unwrap().to_owned(),
                       ~"build",
                       ~"foo"],
                      workspace);
    assert!(target_build_dir(workspace).is_dir());
    assert!(built_executable_exists(workspace, "foo"));
    assert!(fs::readdir(&workspace.join("build")).len() == 1);
}

#[test]
fn test_target_specific_install_dir() {
    let p_id = PkgId::new("foo");
    let workspace = create_local_package(&p_id);
    let workspace = workspace.path();
    let test_sys = test_sysroot();
    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([test_sys.as_str().unwrap().to_owned(),
                       ~"install",
                       ~"foo"],
                      workspace);
    assert!(workspace.join_many([~"lib", host_triple()]).is_dir());
    assert_lib_exists(workspace, &Path::new("foo"), NoVersion);
    assert!(fs::readdir(&workspace.join("lib")).len() == 1);
    assert!(workspace.join("bin").is_dir());
    assert_executable_exists(workspace, "foo");
}

#[test]
#[ignore(reason = "See #7240")]
fn test_dependencies_terminate() {
    let b_id = PkgId::new("b");
    let workspace = create_local_package(&b_id);
    let workspace = workspace.path();
    let b_dir = workspace.join_many(["src", "b-0.0"]);
    let b_subdir = b_dir.join("test");
    fs::mkdir_recursive(&b_subdir, io::UserRWX);
    writeFile(&b_subdir.join("test.rs"),
              "extern mod b; use b::f; #[test] fn g() { f() }");
    command_line_test([~"install", ~"b"], workspace);
}

#[test]
fn install_after_build() {
    let b_id = PkgId::new("b");
    let workspace = create_local_package(&b_id);
    let workspace = workspace.path();
    command_line_test([~"build", ~"b"], workspace);
    command_line_test([~"install", ~"b"], workspace);
    assert_executable_exists(workspace, b_id.short_name);
    assert_lib_exists(workspace, &b_id.path, NoVersion);
}

#[test]
fn reinstall() {
    let b = PkgId::new("b");
    let workspace = create_local_package(&b);
    let workspace = workspace.path();
    // 1. Install, then remove executable file, then install again,
    // and make sure executable was re-installed
    command_line_test([~"install", ~"b"], workspace);
    assert_executable_exists(workspace, b.short_name);
    assert_lib_exists(workspace, &b.path, NoVersion);
    remove_executable_file(&b, workspace);
    command_line_test([~"install", ~"b"], workspace);
    assert_executable_exists(workspace, b.short_name);
    // 2. Build, then remove build executable file, then build again,
    // and make sure executable was re-built.
    command_line_test([~"build", ~"b"], workspace);
    remove_built_executable_file(&b, workspace);
    command_line_test([~"build", ~"b"], workspace);
    assert_built_executable_exists(workspace, b.short_name);
    // 3. Install, then remove both executable and built executable,
    // then install again, make sure both were recreated
    command_line_test([~"install", ~"b"], workspace);
    remove_executable_file(&b, workspace);
    remove_built_executable_file(&b, workspace);
    command_line_test([~"install", ~"b"], workspace);
    assert_executable_exists(workspace, b.short_name);
    assert_built_executable_exists(workspace, b.short_name);
}

#[test]
fn correct_package_name_with_rust_path_hack() {
    /*
    Set rust_path_hack flag

    Try to install bar
    Check that:
    - no output gets produced in any workspace
    - there's an error
    */

    // Set RUST_PATH to something containing only the sources for foo
    let foo_id = PkgId::new("foo");
    let bar_id = PkgId::new("bar");
    let foo_workspace = create_local_package(&foo_id);
    let foo_workspace = foo_workspace.path();
    let dest_workspace = mk_empty_workspace(&Path::new("bar"), &NoVersion, "dest_workspace");
    let dest_workspace = dest_workspace.path();

    writeFile(&dest_workspace.join_many(["src", "bar-0.0", "main.rs"]),
              "extern mod blat; fn main() { let _x = (); }");

    let foo_path = foo_workspace.join_many(["src", "foo-0.0"]);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let rust_path = Some(~[(~"RUST_PATH", format!("{}:{}", dest_workspace.as_str().unwrap(),
                                                  foo_path.as_str().unwrap()))]);
    // bar doesn't exist, but we want to make sure rustpkg doesn't think foo is bar
    command_line_test_expect_fail([~"install", ~"--rust-path-hack", ~"bar"],
                                  // FIXME #3408: Should be NONEXISTENT_PACKAGE_CODE
                               dest_workspace, rust_path, COPY_FAILED_CODE);
    assert!(!executable_exists(dest_workspace, "bar"));
    assert!(!lib_exists(dest_workspace, &bar_id.path.clone(), bar_id.version.clone()));
    assert!(!executable_exists(dest_workspace, "foo"));
    assert!(!lib_exists(dest_workspace, &foo_id.path.clone(), foo_id.version.clone()));
    assert!(!executable_exists(foo_workspace, "bar"));
    assert!(!lib_exists(foo_workspace, &bar_id.path.clone(), bar_id.version.clone()));
    assert!(!executable_exists(foo_workspace, "foo"));
    assert!(!lib_exists(foo_workspace, &foo_id.path.clone(), foo_id.version.clone()));
}

#[test]
fn test_rustpkg_test_creates_exec() {
    let foo_id = PkgId::new("foo");
    let foo_workspace = create_local_package(&foo_id);
    let foo_workspace = foo_workspace.path();
    writeFile(&foo_workspace.join_many(["src", "foo-0.0", "test.rs"]),
              "#[test] fn f() { assert!('a' == 'a'); }");
    command_line_test([~"test", ~"foo"], foo_workspace);
    assert!(test_executable_exists(foo_workspace, "foo"));
}

#[test]
fn test_rustpkg_test_output() {
    let workspace = create_local_package_with_test(&PkgId::new("foo"));
    let output = command_line_test([~"test", ~"foo"], workspace.path());
    let output_str = str::from_utf8(output.output);
    // The first two assertions are separate because test output may
    // contain color codes, which could appear between "test f" and "ok".
    assert!(output_str.contains("test f"));
    assert!(output_str.contains("ok"));
    assert!(output_str.contains("1 passed; 0 failed; 0 ignored; 0 measured"));
}

#[test]
fn test_rustpkg_test_failure_exit_status() {
    let foo_id = PkgId::new("foo");
    let foo_workspace = create_local_package(&foo_id);
    let foo_workspace = foo_workspace.path();
    writeFile(&foo_workspace.join_many(["src", "foo-0.0", "test.rs"]),
              "#[test] fn f() { assert!('a' != 'a'); }");
    let res = command_line_test_partial([~"test", ~"foo"], foo_workspace);
    match res {
        Fail(_) => {},
        Success(..) => fail!("Expected test failure but got success")
    }
}

#[test]
fn test_rustpkg_test_cfg() {
    let foo_id = PkgId::new("foo");
    let foo_workspace = create_local_package(&foo_id);
    let foo_workspace = foo_workspace.path();
    writeFile(&foo_workspace.join_many(["src", "foo-0.0", "test.rs"]),
              "#[test] #[cfg(not(foobar))] fn f() { assert!('a' != 'a'); }");
    let output = command_line_test([~"test", ~"--cfg", ~"foobar", ~"foo"],
                                   foo_workspace);
    let output_str = str::from_utf8(output.output);
    assert!(output_str.contains("0 passed; 0 failed; 0 ignored; 0 measured"));
}

#[test]
fn test_rebuild_when_needed() {
    let foo_id = PkgId::new("foo");
    let foo_workspace = create_local_package(&foo_id);
    let foo_workspace = foo_workspace.path();
    let test_crate = foo_workspace.join_many(["src", "foo-0.0", "test.rs"]);
    writeFile(&test_crate, "#[test] fn f() { assert!('a' == 'a'); }");
    command_line_test([~"test", ~"foo"], foo_workspace);
    assert!(test_executable_exists(foo_workspace, "foo"));
    let test_executable = built_test_in_workspace(&foo_id,
            foo_workspace).expect("test_rebuild_when_needed failed");
    frob_source_file(foo_workspace, &foo_id, "test.rs");
    chmod_read_only(&test_executable);
    match command_line_test_partial([~"test", ~"foo"], foo_workspace) {
        Success(..) => fail!("test_rebuild_when_needed didn't rebuild"),
        Fail(ref r) if r.status.matches_exit_status(65) => (), // ok
        Fail(_) => fail!("test_rebuild_when_needed failed for some other reason")
    }
}

#[test]
#[ignore] // FIXME (#10257): This doesn't work as is since a read only file can't execute
fn test_no_rebuilding() {
    let foo_id = PkgId::new("foo");
    let foo_workspace = create_local_package(&foo_id);
    let foo_workspace = foo_workspace.path();
    let test_crate = foo_workspace.join_many(["src", "foo-0.0", "test.rs"]);
    writeFile(&test_crate, "#[test] fn f() { assert!('a' == 'a'); }");
    command_line_test([~"test", ~"foo"], foo_workspace);
    assert!(test_executable_exists(foo_workspace, "foo"));
    let test_executable = built_test_in_workspace(&foo_id,
                            foo_workspace).expect("test_no_rebuilding failed");
    chmod_read_only(&test_executable);
    match command_line_test_partial([~"test", ~"foo"], foo_workspace) {
        Success(..) => (), // ok
        Fail(ref r) if r.status.matches_exit_status(65) =>
            fail!("test_no_rebuilding failed: it rebuilt the tests"),
        Fail(_) => fail!("test_no_rebuilding failed for some other reason")
    }
}

#[test]
fn test_installed_read_only() {
    // Install sources from a "remote" (actually a local github repo)
    // Check that afterward, sources are read-only and installed under build/
    let mut temp_pkg_id = git_repo_pkg();
    let repo = init_git_repo(&temp_pkg_id.path);
    let repo = repo.path();
    debug!("repo = {}", repo.display());
    let repo_subdir = repo.join_many(["mockgithub.com", "catamorphism", "test-pkg"]);
    debug!("repo_subdir = {}", repo_subdir.display());

    writeFile(&repo_subdir.join("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&repo_subdir.join("lib.rs"),
              "pub fn f() { let _x = (); }");
    add_git_tag(&repo_subdir, ~"0.1"); // this has the effect of committing the files
    // update pkgid to what will be auto-detected
    temp_pkg_id.version = ExactRevision(~"0.1");

    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([~"install", temp_pkg_id.path.as_str().unwrap().to_owned()], repo);

    let ws = repo.join(".rust");
    // Check that all files exist
    debug!("Checking for files in {}", ws.display());
    let exec = target_executable_in_workspace(&temp_pkg_id, &ws);
    debug!("exec = {}", exec.display());
    assert!(exec.exists());
    assert!(is_rwx(&exec));
    let built_lib =
        built_library_in_workspace(&temp_pkg_id,
                                   &ws).expect("test_install_git: built lib should exist");
    assert!(built_lib.exists());
    assert!(is_rwx(&built_lib));

    // Make sure sources are (a) under "build" and (b) read-only
    let src1 = target_build_dir(&ws).join_many([~"src", temp_pkg_id.to_str(), ~"main.rs"]);
    let src2 = target_build_dir(&ws).join_many([~"src", temp_pkg_id.to_str(), ~"lib.rs"]);
    assert!(src1.exists());
    assert!(src2.exists());
    assert!(is_read_only(&src1));
    assert!(is_read_only(&src2));
}

#[test]
fn test_installed_local_changes() {
    let temp_pkg_id = git_repo_pkg();
    let repo = init_git_repo(&temp_pkg_id.path);
    let repo = repo.path();
    debug!("repo = {}", repo.display());
    let repo_subdir = repo.join_many(["mockgithub.com", "catamorphism", "test-pkg"]);
    debug!("repo_subdir = {}", repo_subdir.display());
    fs::mkdir_recursive(&repo.join_many([".rust", "src"]), io::UserRWX);

    writeFile(&repo_subdir.join("main.rs"),
              "fn main() { let _x = (); }");
    writeFile(&repo_subdir.join("lib.rs"),
              "pub fn f() { let _x = (); }");
    add_git_tag(&repo_subdir, ~"0.1"); // this has the effect of committing the files

    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([~"install", temp_pkg_id.path.as_str().unwrap().to_owned()], repo);


    // We installed the dependency.
    // Now start a new workspace and clone it into it
    let hacking_workspace = mk_emptier_workspace("hacking_workspace");
    let hacking_workspace = hacking_workspace.path();
    let target_dir = hacking_workspace.join_many(["src",
                                                  "mockgithub.com",
                                                  "catamorphism",
                                                  "test-pkg-0.0"]);
    debug!("---- git clone {} {}", repo_subdir.display(), target_dir.display());

    let c_res = safe_git_clone(&repo_subdir, &NoVersion, &target_dir);

    match c_res {
        DirToUse(_) => fail!("test_installed_local_changes failed"),
        CheckedOutSources => ()
    };

    // Make a local change to it
    writeFile(&target_dir.join("lib.rs"),
              "pub fn g() { let _x = (); }");

    // Finally, make *another* package that uses it
    let importer_pkg_id = fake_pkg();
    let main_subdir = create_local_package_in(&importer_pkg_id, hacking_workspace);
    writeFile(&main_subdir.join("main.rs"),
              "extern mod test = \"mockgithub.com/catamorphism/test-pkg\"; \
              use test::g;
              fn main() { g(); }");
    // And make sure we can build it

    // FIXME (#9639): This needs to handle non-utf8 paths
    command_line_test([~"build", importer_pkg_id.path.as_str().unwrap().to_owned()],
                      hacking_workspace);
}

#[test]
fn test_7402() {
    let dir = create_local_package(&PkgId::new("foo"));
    let dest_workspace = TempDir::new("more_rust").expect("test_7402");
    let dest_workspace = dest_workspace.path();
    // FIXME (#9639): This needs to handle non-utf8 paths
    let rust_path = Some(~[(~"RUST_PATH",
                            format!("{}:{}", dest_workspace.as_str().unwrap(),
                                    dir.path().as_str().unwrap()))]);
    let cwd = os::getcwd();
    command_line_test_with_env([~"install", ~"foo"], &cwd, rust_path);
    assert_executable_exists(dest_workspace, "foo");
}

#[test]
fn test_compile_error() {
    let foo_id = PkgId::new("foo");
    let foo_workspace = create_local_package(&foo_id);
    let foo_workspace = foo_workspace.path();
    let main_crate = foo_workspace.join_many(["src", "foo-0.0", "main.rs"]);
    // Write something bogus
    writeFile(&main_crate, "pub fn main() { if 42 != ~\"the answer\" { fail!(); } }");
    let result = command_line_test_partial([~"build", ~"foo"], foo_workspace);
    match result {
        Success(..) => fail!("Failed by succeeding!"), // should be a compile error
        Fail(ref status) => {
            debug!("Failed with status {:?}... that's good, right?", status);
        }
    }
}

#[test]
fn find_sources_in_cwd() {
    let temp_dir = TempDir::new("sources").expect("find_sources_in_cwd failed");
    let temp_dir = temp_dir.path();
    let source_dir = temp_dir.join("foo");
    fs::mkdir_recursive(&source_dir, io::UserRWX);
    writeFile(&source_dir.join("main.rs"),
              "fn main() { let _x = (); }");
    command_line_test([~"install", ~"foo"], &source_dir);
    assert_executable_exists(&source_dir.join(".rust"), "foo");
}

#[test]
#[ignore(reason="busted")]
fn test_c_dependency_ok() {
    // Pkg has a custom build script that adds a single C file as a dependency, and
    // registers a hook to build it if it's not fresh
    // After running `build`, test that the C library built

    let dir = create_local_package(&PkgId::new("cdep"));
    let dir = dir.path();
    writeFile(&dir.join_many(["src", "cdep-0.0", "main.rs"]),
              "#[link_args = \"-lfoo\"]\nextern { fn f(); } \
              \nfn main() { unsafe { f(); } }");
    writeFile(&dir.join_many(["src", "cdep-0.0", "foo.c"]), "void f() {}");

    debug!("dir = {}", dir.display());
    let source = Path::new(file!()).dir_path().join_many(
        [~"testsuite", ~"pass", ~"src", ~"c-dependencies", ~"pkg.rs"]);
    fs::copy(&source, &dir.join_many([~"src", ~"cdep-0.0", ~"pkg.rs"]));
    command_line_test([~"build", ~"cdep"], dir);
    assert_executable_exists(dir, "cdep");
    let out_dir = target_build_dir(dir).join("cdep");
    let c_library_path = out_dir.join(platform_library_name("foo"));
    debug!("c library path: {}", c_library_path.display());
    assert!(c_library_path.exists());
}

#[ignore(reason="rustpkg is not reentrant")]
#[test]
#[ignore(reason="busted")]
fn test_c_dependency_no_rebuilding() {
    let dir = create_local_package(&PkgId::new("cdep"));
    let dir = dir.path();
    writeFile(&dir.join_many(["src", "cdep-0.0", "main.rs"]),
              "#[link_args = \"-lfoo\"]\nextern { fn f(); } \
              \nfn main() { unsafe { f(); } }");
    writeFile(&dir.join_many(["src", "cdep-0.0", "foo.c"]), "void f() {}");

    debug!("dir = {}", dir.display());
    let source = Path::new(file!()).dir_path().join_many(
        [~"testsuite", ~"pass", ~"src", ~"c-dependencies", ~"pkg.rs"]);
    fs::copy(&source, &dir.join_many([~"src", ~"cdep-0.0", ~"pkg.rs"]));
    command_line_test([~"build", ~"cdep"], dir);
    assert_executable_exists(dir, "cdep");
    let out_dir = target_build_dir(dir).join("cdep");
    let c_library_path = out_dir.join(platform_library_name("foo"));
    debug!("c library path: {}", c_library_path.display());
    assert!(c_library_path.exists());

    // Now, make it read-only so rebuilding will fail
    assert!(chmod_read_only(&c_library_path));

    match command_line_test_partial([~"build", ~"cdep"], dir) {
        Success(..) => (), // ok
        Fail(ref r) if r.status.matches_exit_status(65) =>
            fail!("test_c_dependency_no_rebuilding failed: \
                    it tried to rebuild foo.c"),
        Fail(_) =>
            fail!("test_c_dependency_no_rebuilding failed for some other reason")
    }
}

#[test]
#[ignore(reason="busted")]
fn test_c_dependency_yes_rebuilding() {
    let dir = create_local_package(&PkgId::new("cdep"));
    let dir = dir.path();
    writeFile(&dir.join_many(["src", "cdep-0.0", "main.rs"]),
              "#[link_args = \"-lfoo\"]\nextern { fn f(); } \
              \nfn main() { unsafe { f(); } }");
    let c_file_name = dir.join_many(["src", "cdep-0.0", "foo.c"]);
    writeFile(&c_file_name, "void f() {}");

    let source = Path::new(file!()).dir_path().join_many(
        [~"testsuite", ~"pass", ~"src", ~"c-dependencies", ~"pkg.rs"]);
    let target = dir.join_many([~"src", ~"cdep-0.0", ~"pkg.rs"]);
    debug!("Copying {} -> {}", source.display(), target.display());
    fs::copy(&source, &target);
    command_line_test([~"build", ~"cdep"], dir);
    assert_executable_exists(dir, "cdep");
    let out_dir = target_build_dir(dir).join("cdep");
    let c_library_path = out_dir.join(platform_library_name("foo"));
    debug!("c library path: {}", c_library_path.display());
    assert!(c_library_path.exists());

    // Now, make the Rust library read-only so rebuilding will fail
    match built_library_in_workspace(&PkgId::new("cdep"), dir) {
        Some(ref pth) => assert!(chmod_read_only(pth)),
        None => assert_built_library_exists(dir, "cdep")
    }

    match command_line_test_partial([~"build", ~"cdep"], dir) {
        Success(..) => fail!("test_c_dependency_yes_rebuilding failed: \
                            it didn't rebuild and should have"),
        Fail(ref r) if r.status.matches_exit_status(65) => (),
        Fail(_) => fail!("test_c_dependency_yes_rebuilding failed for some other reason")
    }
}

// n.b. This might help with #10253, or at least the error will be different.
#[test]
fn correct_error_dependency() {
    // Supposing a package we're trying to install via a dependency doesn't
    // exist, we should throw a condition, and not ICE
    let workspace_dir = create_local_package(&PkgId::new("badpkg"));

    let dir = workspace_dir.path();
    let main_rs = dir.join_many(["src", "badpkg-0.0", "main.rs"]);
    writeFile(&main_rs,
              "extern mod p = \"some_package_that_doesnt_exist\";
               fn main() {}");
    match command_line_test_partial([~"build", ~"badpkg"], dir) {
        Fail(ProcessOutput{ error: error, output: output, .. }) => {
            assert!(str::is_utf8(error));
            assert!(str::is_utf8(output));
            let error_str = str::from_utf8(error);
            let out_str   = str::from_utf8(output);
            debug!("ss = {}", error_str);
            debug!("out_str = {}", out_str);
            if out_str.contains("Package badpkg depends on some_package_that_doesnt_exist") &&
                !error_str.contains("nonexistent_package") {
                // Ok
                ()
            } else {
                fail!("Wrong error");
            }
        }
        Success(..)       => fail!("Test passed when it should have failed")
    }
}

/// Returns true if p exists and is executable
fn is_executable(p: &Path) -> bool {
    p.exists() && p.stat().perm & io::UserExecute == io::UserExecute
}
