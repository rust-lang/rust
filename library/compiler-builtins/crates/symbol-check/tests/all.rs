use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use assert_cmd::assert::Assert;
use assert_cmd::cargo::cargo_bin_cmd;
use tempfile::tempdir;

trait AssertExt {
    fn stderr_contains(self, s: &str) -> Self;
}

impl AssertExt for Assert {
    #[track_caller]
    fn stderr_contains(self, s: &str) -> Self {
        let out = String::from_utf8_lossy(&self.get_output().stderr);
        assert!(out.contains(s), "looking for: `{s}`\nout:\n```\n{out}\n```");
        self
    }
}

#[test]
fn test_duplicates() {
    let t = TestTarget::from_env();
    let dir = tempdir().unwrap();
    let dup_out = dir.path().join("dup.o");
    let lib_out = dir.path().join("libfoo.rlib");

    // For the "bad" file, we need duplicate symbols from different object files in the archive. Do
    // this reliably by building an archive and a separate object file then merging them.
    t.rustc_build(&input_dir().join("duplicates.rs"), &lib_out, |cmd| cmd);
    t.rustc_build(&input_dir().join("duplicates.rs"), &dup_out, |cmd| {
        cmd.arg("--emit=obj")
    });

    let mut ar = t.cc_build().get_archiver();

    if ar.get_program().to_string_lossy().contains("lib.exe") {
        let mut out_arg = OsString::from("-out:");
        out_arg.push(&lib_out);
        ar.arg(&out_arg);
        // Repeating the same file as the first arg makes lib.exe append (taken from the
        // `cc` implementation).
        ar.arg(&lib_out);
    } else {
        ar.arg("rs")
            // Eat an `libfoo.rlib(lib.rmeta) has no symbols` info message on MacOS
            .stderr(Stdio::null())
            .arg(&lib_out);
    }

    run(ar.arg(&dup_out));

    let assert = t.symcheck_exe().arg(&lib_out).assert();
    assert
        .failure()
        .stderr_contains("duplicate symbols")
        .stderr_contains("FDUP")
        .stderr_contains("IDUP")
        .stderr_contains("fndup");
}

#[test]
fn test_core_symbols() {
    let t = TestTarget::from_env();
    let dir = tempdir().unwrap();
    let lib_out = dir.path().join("libfoo.rlib");
    t.rustc_build(&input_dir().join("core_symbols.rs"), &lib_out, |cmd| cmd);
    let assert = t.symcheck_exe().arg(&lib_out).assert();
    assert
        .failure()
        .stderr_contains("found 1 undefined symbols from core")
        .stderr_contains("from_utf8");
}

#[test]
fn test_good_lib() {
    let t = TestTarget::from_env();
    let dir = tempdir().unwrap();
    let lib_out = dir.path().join("libfoo.rlib");
    t.rustc_build(&input_dir().join("good.rs"), &lib_out, |cmd| cmd);
    let assert = t.symcheck_exe().arg(&lib_out).assert();
    assert.success();
}

/// Since symcheck is a hostprog, the target we want to build and test symcheck for may not be the
/// same as the host target.
struct TestTarget {
    triple: String,
}

impl TestTarget {
    fn from_env() -> Self {
        let triple = match env::var("SYMCHECK_TEST_TARGET") {
            Ok(t) => t,
            // Require on CI so we don't accidentally always test the native target
            _ if env::var("CI").is_ok() => panic!("SYMCHECK_TEST_TARGET must be set in CI"),
            // Fall back to native for local convenience.
            Err(_) => env!("HOST").to_string(),
        };

        println!("using target {triple}");
        Self { triple }
    }

    /// Build i -> o with optional additional configuration.
    fn rustc_build(&self, i: &Path, o: &Path, mut f: impl FnMut(&mut Command) -> &mut Command) {
        let mut cmd = Command::new("rustc");
        cmd.arg(i)
            .arg("--target")
            .arg(&self.triple)
            .arg("--crate-type=lib")
            .arg("-o")
            .arg(o);
        f(&mut cmd);
        run(&mut cmd);
    }

    /// Configure `cc` with the host and target.
    fn cc_build(&self) -> cc::Build {
        let mut b = cc::Build::new();
        b.host(env!("HOST"))
            .target(&self.triple)
            .opt_level(0)
            .cargo_debug(true)
            .cargo_metadata(false);
        b
    }

    fn symcheck_exe(&self) -> assert_cmd::Command {
        let mut cmd = cargo_bin_cmd!();
        cmd.arg("--check");
        cmd
    }
}

fn input_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/input")
}

#[track_caller]
fn run(cmd: &mut Command) {
    eprintln!("+ {cmd:?}");
    let out = cmd.output().unwrap();
    println!("{}", String::from_utf8_lossy(&out.stdout));
    eprintln!("{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{:?}", out.status);
}
