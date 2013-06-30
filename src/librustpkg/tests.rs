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
use std::{io, libc, os, result, run, str, vec};
use extra::tempfile::mkdtemp;
use std::run::ProcessOutput;
use package_path::*;
use package_id::{PkgId};
use package_source::*;
use version::{ExactRevision, NoVersion, Version};
use path_util::{target_executable_in_workspace, target_library_in_workspace,
               target_test_in_workspace, target_bench_in_workspace,
               make_dir_rwx, u_rwx, library_in_workspace,
               built_bench_in_workspace, built_test_in_workspace,
               built_library_in_workspace, built_executable_in_workspace,
                installed_library_in_workspace, rust_path};
use target::*;

/// Returns the last-modified date as an Option
fn datestamp(p: &Path) -> Option<libc::time_t> {
    p.stat().map(|stat| stat.st_mtime)
}

fn fake_ctxt(sysroot_opt: Option<@Path>) -> Ctx {
    Ctx {
        sysroot_opt: sysroot_opt,
        json: false,
        dep_cache: @mut HashMap::new()
    }
}

fn fake_pkg() -> PkgId {
    let sn = ~"bogus";
    let remote = RemotePath(Path(sn));
    PkgId {
        local_path: normalize(copy remote),
        remote_path: remote,
        short_name: sn,
        version: NoVersion
    }
}

fn remote_pkg() -> PkgId {
    let remote = RemotePath(Path("github.com/catamorphism/test-pkg"));
    PkgId {
        local_path: normalize(copy remote),
        remote_path: remote,
        short_name: ~"test_pkg",
        version: NoVersion
    }
}

fn writeFile(file_path: &Path, contents: &str) {
    let out: @io::Writer =
        result::get(&io::file_writer(file_path,
                                     [io::Create, io::Truncate]));
    out.write_line(contents);
}

fn mk_empty_workspace(short_name: &LocalPath, version: &Version) -> Path {
    let workspace_dir = mkdtemp(&os::tmpdir(), "test").expect("couldn't create temp dir");
    mk_workspace(&workspace_dir, short_name, version);
    workspace_dir
}

fn mk_workspace(workspace: &Path, short_name: &LocalPath, version: &Version) -> Path {
    // include version number in directory name
    let package_dir = workspace.push("src").push(fmt!("%s-%s",
                                                      short_name.to_str(), version.to_str()));
    assert!(os::mkdir_recursive(&package_dir, u_rwx));
    package_dir
}

fn mk_temp_workspace(short_name: &LocalPath, version: &Version) -> Path {
    let package_dir = mk_empty_workspace(short_name,
                                         version).push("src").push(fmt!("%s-%s",
                                                            short_name.to_str(),
                                                            version.to_str()));

    debug!("Created %s and does it exist? %?", package_dir.to_str(),
          os::path_is_dir(&package_dir));
    // Create main, lib, test, and bench files
    debug!("mk_workspace: creating %s", package_dir.to_str());
    assert!(os::mkdir_recursive(&package_dir, u_rwx));
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

/// Runs `rustpkg` (based on the directory that this executable was
/// invoked from) with the given arguments, in the given working directory.
/// Returns the process's output.
fn command_line_test(args: &[~str], cwd: &Path) -> ProcessOutput {
    let cmd = test_sysroot().push("bin").push("rustpkg").to_str();
    let cwd = normalize(RemotePath(copy *cwd));
    debug!("About to run command: %? %? in %s", cmd, args, cwd.to_str());
    assert!(os::path_is_dir(&*cwd));
    let mut prog = run::Process::new(cmd, args, run::ProcessOptions { env: None,
                                                           dir: Some(&*cwd),
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
        fail!("Command %s %? failed with exit code %?",
              cmd, args, output.status);
    }
    output
}

fn make_git_repo(short_name: &str) -> Path {
    let temp_d = mk_temp_workspace(&normalize(RemotePath(Path(short_name))), &NoVersion);
    debug!("Dry run: would initialize %s as a git repository", temp_d.pop().pop().to_str());
    temp_d.pop().pop()
}

fn add_git_tag(repo: &Path, tag: &str) {
    debug!("Dry run: would add tag %s to repo %s", tag, repo.to_str());
}

fn create_local_package(pkgid: &PkgId) -> Path {
    let parent_dir = mk_temp_workspace(&pkgid.local_path, &pkgid.version);
    debug!("Created empty package dir for %s, returning %s", pkgid.to_str(), parent_dir.to_str());
    parent_dir.pop().pop()
}

fn create_local_package_in(pkgid: &PkgId, pkgdir: &Path) -> Path {

    let package_dir = pkgdir.push("src").push(pkgid.to_str());

    // Create main, lib, test, and bench files
    assert!(os::mkdir_recursive(&package_dir, u_rwx));
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

fn assert_lib_exists(repo: &Path, short_name: &str) {
    debug!("assert_lib_exists: repo = %s, short_name = %s", repo.to_str(), short_name);
    let lib = target_library_in_workspace(&PkgId::new(short_name), repo);
    assert!(os::path_exists(&lib));
    assert!(is_rwx(&lib));
}

fn assert_executable_exists(repo: &Path, short_name: &str) {
    debug!("assert_executable_exists: repo = %s, short_name = %s", repo.to_str(), short_name);
    let exec = target_executable_in_workspace(&PkgId::new(short_name), repo);
    assert!(os::path_exists(&exec));
    assert!(is_rwx(&exec));
}

fn command_line_test_output(args: &[~str]) -> ~[~str] {
    let mut result = ~[];
    let p_output = command_line_test(args, &os::getcwd());
    let test_output = str::from_bytes(p_output.output);
    for test_output.split_iter('\n').advance |s| {
        result.push(s.to_owned());
    }
    result
}

// assumes short_name and local_path are one and the same -- I should fix
fn lib_output_file_name(workspace: &Path, parent: &str, short_name: &str) -> Path {
    debug!("lib_output_file_name: given %s and parent %s and short name %s",
           workspace.to_str(), parent, short_name);
    library_in_workspace(&normalize(RemotePath(Path(short_name))),
                         short_name,
                         Build,
                         workspace,
                         "build").expect("lib_output_file_name")
}

fn output_file_name(workspace: &Path, short_name: &str) -> Path {
    workspace.push(fmt!("%s%s", short_name, os::EXE_SUFFIX))
}

fn touch_source_file(workspace: &Path, pkgid: &PkgId) {
    use conditions::bad_path::cond;
    let pkg_src_dir = workspace.push("src").push(pkgid.to_str());
    let contents = os::list_dir_path(&pkg_src_dir);
    for contents.iter().advance |p| {
        if p.filetype() == Some(~".rs") {
            // should be able to do this w/o a process
            if run::process_output("touch", [p.to_str()]).status != 0 {
                let _ = cond.raise((copy pkg_src_dir, ~"Bad path"));
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
    for contents.iter().advance |p| {
        if p.filetype() == Some(~".rs") {
            maybe_p = Some(p);
            break;
        }
    }
    match maybe_p {
        Some(p) => {
            let w = io::file_writer(*p, &[io::Append]);
            match w {
                Err(s) => { let _ = cond.raise((copy **p, fmt!("Bad path: %s", s))); }
                Ok(w)  => w.write_line("")
            }
        }
        None => fail!(fmt!("frob_source_file failed to find a source file in %s",
                           pkg_src_dir.to_str()))
    }
}

#[test] #[ignore] //FIXME(#7249)
fn test_all() {
    // FIXME(#7071): these tests use rustc, so they can't be run in parallel
    //               until this issue is resolved
    test_make_dir_rwx();
    test_install_valid();
    test_install_invalid();
    test_install_url();
    test_package_ids_must_be_relative_path_like();
    test_package_version();
}

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

fn test_install_valid() {
    use path_util::installed_library_in_workspace;

    let sysroot = test_sysroot();
    debug!("sysroot = %s", sysroot.to_str());
    let ctxt = fake_ctxt(Some(@sysroot));
    let temp_pkg_id = fake_pkg();
    let temp_workspace = mk_temp_workspace(&temp_pkg_id.local_path, &NoVersion).pop().pop();
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
            copy temp_workspace
        }).in {
            ctxt.install(&temp_workspace, &pkgid);
        }
    }
    assert!(error_occurred && error1_occurred);
}

fn test_install_url() {
    let workspace = mkdtemp(&os::tmpdir(), "test").expect("couldn't create temp dir");
    let sysroot = test_sysroot();
    debug!("sysroot = %s", sysroot.to_str());
    let ctxt = fake_ctxt(Some(@sysroot));
    let temp_pkg_id = remote_pkg();
    // should have test, bench, lib, and main
    ctxt.install(&workspace, &temp_pkg_id);
    // Check that all files exist
    let exec = target_executable_in_workspace(&temp_pkg_id, &workspace);
    debug!("exec = %s", exec.to_str());
    assert!(os::path_exists(&exec));
    assert!(is_rwx(&exec));
    let _built_lib =
        built_library_in_workspace(&temp_pkg_id,
                                   &workspace).expect("test_install_url: built lib should exist");
    let lib = target_library_in_workspace(&temp_pkg_id, &workspace);
    debug!("lib = %s", lib.to_str());
    assert!(os::path_exists(&lib));
    assert!(is_rwx(&lib));
    let built_test = built_test_in_workspace(&temp_pkg_id,
                         &workspace).expect("test_install_url: built test should exist");
    assert!(os::path_exists(&built_test));
    let built_bench = built_bench_in_workspace(&temp_pkg_id,
                          &workspace).expect("test_install_url: built bench should exist");
    assert!(os::path_exists(&built_bench));
    // And that the test and bench executables aren't installed
    let test = target_test_in_workspace(&temp_pkg_id, &workspace);
    assert!(!os::path_exists(&test));
    debug!("test = %s", test.to_str());
    let bench = target_bench_in_workspace(&temp_pkg_id, &workspace);
    debug!("bench = %s", bench.to_str());
    assert!(!os::path_exists(&bench));
}

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
    assert!("github.com/catamorphism/test_pkg-0.1" ==
            PkgId::new("github.com/catamorphism/test-pkg").to_str());

    do cond.trap(|(p, e)| {
        assert!("" == p.to_str());
        assert!("0-length pkgid" == e);
        copy whatever
    }).in {
        let x = PkgId::new("");
        assert_eq!(~"foo-0.1", x.to_str());
    }

    do cond.trap(|(p, e)| {
        assert_eq!(p.to_str(), os::make_absolute(&Path("foo/bar/quux")).to_str());
        assert!("absolute pkgid" == e);
        copy whatever
    }).in {
        let z = PkgId::new(os::make_absolute(&Path("foo/bar/quux")).to_str());
        assert_eq!(~"foo-0.1", z.to_str());
    }

}

fn test_package_version() {
    let temp_pkg_id = PkgId::new("github.com/catamorphism/test_pkg_version");
    match temp_pkg_id.version {
        ExactRevision(~"0.4") => (),
        _ => fail!(fmt!("test_package_version: package version was %?, expected Some(0.4)",
                        temp_pkg_id.version))
    }
    let temp = mk_empty_workspace(&LocalPath(Path("test_pkg_version")), &temp_pkg_id.version);
    let ctx = fake_ctxt(Some(@test_sysroot()));
    ctx.build(&temp, &temp_pkg_id);
    assert!(match built_library_in_workspace(&temp_pkg_id, &temp) {
        Some(p) => p.to_str().ends_with(fmt!("0.4%s", os::consts::DLL_SUFFIX)),
        None    => false
    });
    assert!(built_executable_in_workspace(&temp_pkg_id, &temp)
            == Some(temp.push("build").
                    push("github.com").
                    push("catamorphism").
                    push("test_pkg_version").
                    push("test_pkg_version")));
}

// FIXME #7006: Fails on linux for some reason
#[test]
#[ignore]
fn test_package_request_version() {
    let temp_pkg_id = PkgId::new("github.com/catamorphism/test_pkg_version#0.3");
    let temp = mk_empty_workspace(&LocalPath(Path("test_pkg_version")), &ExactRevision(~"0.3"));
    let pkg_src = PkgSrc::new(&temp, &temp, &temp_pkg_id);
    match temp_pkg_id.version {
        ExactRevision(~"0.3") => {
            match pkg_src.fetch_git() {
                Some(p) => {
                    assert!(os::path_exists(&p.push("version-0.3-file.txt")));
                    assert!(!os::path_exists(&p.push("version-0.4-file.txt")));

                }
                None => fail!("test_package_request_version: fetch_git failed")
            }
        }
        ExactRevision(n) => {
            fail!("n is %? and %? %s %?", n, n, if n == ~"0.3" { "==" } else { "!=" }, "0.3");
        }
        _ => fail!(fmt!("test_package_version: package version was %?, expected ExactRevision(0.3)",
                        temp_pkg_id.version))
    }
    let c = fake_ctxt(Some(@test_sysroot()));
    c.install(&temp, &temp_pkg_id);
    debug!("installed_library_in_workspace(%s, %s) = %?", temp_pkg_id.short_name, temp.to_str(),
           installed_library_in_workspace(temp_pkg_id.short_name, &temp));
    assert!(match installed_library_in_workspace(temp_pkg_id.short_name, &temp) {
        Some(p) => {
            debug!("installed: %s", p.to_str());
            p.to_str().ends_with(fmt!("0.3%s", os::consts::DLL_SUFFIX))
        }
        None    => false
    });
    assert!(target_executable_in_workspace(&temp_pkg_id, &temp)
            == temp.push("bin").push("test_pkg_version"));

}

// Tests above should (maybe) be converted to shell out to rustpkg, too

#[test]
#[ignore (reason = "http-client not ported to rustpkg yet")]
fn rustpkg_install_url_2() {
    let temp_dir = mkdtemp(&os::tmpdir(), "rustpkg_install_url_2").expect("rustpkg_install_url_2");
    command_line_test([~"install", ~"github.com/mozilla-servo/rust-http-client"],
                     &temp_dir);
}

#[test]
fn rustpkg_library_target() {
    let foo_repo = make_git_repo("foo");
    add_git_tag(&foo_repo, "1.0");
    command_line_test([~"install", ~"foo"], &foo_repo);
    assert_lib_exists(&foo_repo, "foo");
}

#[test]
fn rustpkg_local_pkg() {
    let dir = create_local_package(&PkgId::new("foo"));
    command_line_test([~"install", ~"foo"], &dir);
    assert_executable_exists(&dir, "foo");
}

#[test]
#[ignore] // XXX Failing on dist-linux bot
fn package_script_with_default_build() {
    let dir = create_local_package(&PkgId::new("fancy-lib"));
    debug!("dir = %s", dir.to_str());
    let source = test_sysroot().pop().pop().pop().push("src").push("librustpkg").
        push("testsuite").push("pass").push("src").push("fancy-lib").push("pkg.rs");
    debug!("package_script_with_default_build: %s", source.to_str());
    if !os::copy_file(&source,
                      & dir.push("src").push("fancy_lib-0.1").push("pkg.rs")) {
        fail!("Couldn't copy file");
    }
    command_line_test([~"install", ~"fancy-lib"], &dir);
    assert_lib_exists(&dir, "fancy-lib");
    assert!(os::path_exists(&dir.push("build").push("fancy_lib").push("generated.rs")));
}

#[test]
#[ignore (reason = "Un-ignore when #7071 is fixed")]
fn rust_path_test() {
    let dir_for_path = mkdtemp(&os::tmpdir(), "more_rust").expect("rust_path_test failed");
    let dir = mk_workspace(&dir_for_path, &normalize(RemotePath(Path("foo"))), &NoVersion);
    debug!("dir = %s", dir.to_str());
    writeFile(&Path("/Users/tjc/more_rust/src/foo-0.1/main.rs"),
              "fn main() { let _x = (); }");

    let cwd = os::getcwd();
    debug!("cwd = %s", cwd.to_str());
    let mut prog = run::Process::new("rustpkg",
                                     [~"install", ~"foo"],
                                     run::ProcessOptions { env: Some(&[(~"RUST_PATH",
                                                                       dir_for_path.to_str())]),
                                                          dir: Some(&cwd),
                                                          in_fd: None,
                                                          out_fd: None,
                                                          err_fd: None
                                                         });
    prog.finish_with_output();
    assert_executable_exists(&dir_for_path, "foo");
}

#[test]
fn rust_path_contents() {
    let dir = mkdtemp(&os::tmpdir(), "rust_path").expect("rust_path_contents failed");
    let abc = &dir.push("A").push("B").push("C");
    assert!(os::mkdir_recursive(&abc.push(".rust"), u_rwx));
    assert!(os::mkdir_recursive(&abc.pop().push(".rust"), u_rwx));
    assert!(os::mkdir_recursive(&abc.pop().pop().push(".rust"), u_rwx));
    assert!(do os::change_dir_locked(&dir.push("A").push("B").push("C")) {
        let p = rust_path();
        let cwd = os::getcwd().push(".rust");
        let parent = cwd.pop().pop().push(".rust");
        let grandparent = cwd.pop().pop().pop().push(".rust");
        assert!(vec::contains(p, &cwd));
        assert!(vec::contains(p, &parent));
        assert!(vec::contains(p, &grandparent));
        for p.iter().advance() |a_path| {
            assert!(!a_path.components.is_empty());
        }
    });
}

#[test]
fn rust_path_parse() {
    os::setenv("RUST_PATH", "/a/b/c:/d/e/f:/g/h/i");
    let paths = rust_path();
    assert!(vec::contains(paths, &Path("/g/h/i")));
    assert!(vec::contains(paths, &Path("/d/e/f")));
    assert!(vec::contains(paths, &Path("/a/b/c")));
    os::unsetenv("RUST_PATH");
}

#[test]
#[ignore(reason = "Package database not yet implemented")]
fn install_remove() {
    let foo = PkgId::new("foo");
    let bar = PkgId::new("bar");
    let quux = PkgId::new("quux");
    let dir = mkdtemp(&os::tmpdir(), "install_remove").expect("install_remove");
    create_local_package_in(&foo, &dir);
    create_local_package_in(&bar, &dir);
    create_local_package_in(&quux, &dir);
    command_line_test([~"install", ~"foo"], &dir);
    command_line_test([~"install", ~"bar"], &dir);
    command_line_test([~"install", ~"quux"], &dir);
    let list_output = command_line_test_output([~"list"]);
    assert!(list_output.iter().any_(|x| x == &~"foo"));
    assert!(list_output.iter().any_(|x| x == &~"bar"));
    assert!(list_output.iter().any_(|x| x == &~"quux"));
    command_line_test([~"remove", ~"foo"], &dir);
    let list_output = command_line_test_output([~"list"]);
    assert!(!list_output.iter().any_(|x| x == &~"foo"));
    assert!(list_output.iter().any_(|x| x == &~"bar"));
    assert!(list_output.iter().any_(|x| x == &~"quux"));
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

#[test]
fn do_rebuild_dep_dates_change() {
    let p_id = PkgId::new("foo");
    let dep_id = PkgId::new("bar");
    let workspace = create_local_package_with_dep(&p_id, &dep_id);
    command_line_test([~"build", ~"foo"], &workspace);
    let bar_date = datestamp(&lib_output_file_name(&workspace, "build", "bar"));
    touch_source_file(&workspace, &dep_id);
    command_line_test([~"build", ~"foo"], &workspace);
    let new_bar_date = datestamp(&lib_output_file_name(&workspace, "build", "bar"));
    assert!(new_bar_date > bar_date);
}

#[test]
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
#[ignore(reason = "list not yet implemented")]
fn test_versions() {
    let workspace = create_local_package(&PkgId::new("foo#0.1"));
    create_local_package(&PkgId::new("foo#0.2"));
    command_line_test([~"install", ~"foo#0.1"], &workspace);
    let output = command_line_test_output([~"list"]);
    // make sure output includes versions
    assert!(!output.iter().any_(|x| x == &~"foo#0.2"));
}

#[test]
#[ignore(reason = "do not yet implemented")]
fn test_build_hooks() {
    let workspace = create_local_package_with_custom_build_hook(&PkgId::new("foo"), "frob");
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
#[ignore(reason = "uninstall not yet implemented")]
fn test_uninstall() {
    let workspace = create_local_package(&PkgId::new("foo"));
    let _output = command_line_test([~"info", ~"foo"], &workspace);
    command_line_test([~"uninstall", ~"foo"], &workspace);
    let output = command_line_test([~"list"], &workspace);
    assert!(!str::from_bytes(output.output).contains("foo"));
}
