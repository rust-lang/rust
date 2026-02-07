use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::LazyLock;

use assert_cmd::assert::Assert;
use assert_cmd::cargo::cargo_bin_cmd;
use tempfile::tempdir;

trait AssertExt {
    fn stderr_contains(self, s: &str) -> Self;
}

impl AssertExt for Assert {
    fn stderr_contains(self, s: &str) -> Self {
        let out = String::from_utf8_lossy(&self.get_output().stderr);
        assert!(out.contains(s), "looking for: `{s}`\nout:\n```\n{out}\n```");
        self
    }
}

#[test]
fn test_duplicates() {
    let dir = tempdir().unwrap();
    let dup_out = dir.path().join("dup.o");
    let lib_out = dir.path().join("libfoo.rlib");

    // For the "bad" file, we need duplicate symbols from different object files in the archive. Do
    // this reliably by building an archive and a separate object file then merging them.
    rustc_build(&input_dir().join("duplicates.rs"), &lib_out, |cmd| cmd);
    rustc_build(&input_dir().join("duplicates.rs"), &dup_out, |cmd| {
        cmd.arg("--emit=obj")
    });

    let mut ar = cc_build().get_archiver();

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
    let status = ar.arg(&dup_out).status().unwrap();
    assert!(status.success());

    let assert = cargo_bin_cmd!().arg("--check").arg(&lib_out).assert();
    assert
        .failure()
        .stderr_contains("duplicate symbols")
        .stderr_contains("FDUP")
        .stderr_contains("IDUP")
        .stderr_contains("fndup");
}

#[test]
fn test_core_symbols() {
    let dir = tempdir().unwrap();
    let lib_out = dir.path().join("libfoo.rlib");
    rustc_build(&input_dir().join("core_symbols.rs"), &lib_out, |cmd| cmd);
    let assert = cargo_bin_cmd!().arg("--check").arg(&lib_out).assert();
    assert
        .failure()
        .stderr_contains("found 1 undefined symbols from core")
        .stderr_contains("from_utf8");
}

#[test]
fn test_good() {
    let dir = tempdir().unwrap();
    let lib_out = dir.path().join("libfoo.rlib");
    rustc_build(&input_dir().join("good.rs"), &lib_out, |cmd| cmd);
    let assert = cargo_bin_cmd!().arg("--check").arg(&lib_out).assert();
    assert.success();
}

/// Build i -> o with optional additional configuration.
fn rustc_build(i: &Path, o: &Path, mut f: impl FnMut(&mut Command) -> &mut Command) {
    let mut cmd = Command::new("rustc");
    cmd.arg(i)
        .arg("--target")
        .arg(target())
        .arg("--crate-type=lib")
        .arg("-o")
        .arg(o);
    f(&mut cmd);
    let status = cmd.status().unwrap();
    assert!(status.success());
}

/// Configure `cc` with the host and target.
fn cc_build() -> cc::Build {
    let mut b = cc::Build::new();
    b.host(env!("HOST")).target(&target());
    b
}

/// Symcheck runs on the host but we want to verify that we find issues on all targets, so
/// the cross target may be specified.
fn target() -> String {
    static TARGET: LazyLock<String> = LazyLock::new(|| {
        let target = match env::var("SYMCHECK_TEST_TARGET") {
            Ok(t) => t,
            // Require on CI so we don't accidentally always test the native target
            _ if env::var("CI").is_ok() => panic!("SYMCHECK_TEST_TARGET must be set in CI"),
            // Fall back to native for local convenience.
            Err(_) => env!("HOST").to_string(),
        };

        println!("using target {target}");
        target
    });

    TARGET.clone()
}

fn input_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/input")
}
