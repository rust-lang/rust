//! This test checks if Internal Compilation Error (ICE) dump files `rustc-ice*.txt` work as
//! expected.
//!
//! - Basic sanity checks on a default ICE dump.
//! - Get the number of lines from the dump files without any `RUST_BACKTRACE` options, then check
//!   ICE dump file (line count) is not affected by `RUSTC_BACKTRACE` settings.
//! - Check that disabling ICE dumping results in zero dump files created.
//! - Check that the ICE dump contain some of the expected strings.
//! - Check that `RUST_BACKTRACE=0` prevents ICE dump from created.
//! - Exercise the `-Zmetrics-dir` nightly flag (#128914):
//!     - When `-Zmetrics=dir=PATH` is present but `RUSTC_ICE` is not set, check that the ICE dump
//!       is placed under `PATH`.
//!     - When `RUSTC_ICE=RUSTC_ICE_PATH` and `-Zmetrics-dir=METRICS_PATH` are both provided, check
//!       that `RUSTC_ICE_PATH` takes precedence and no ICE dump is emitted under `METRICS_PATH`.
//!
//! See <https://github.com/rust-lang/rust/pull/108714>.
//!
//! # Test history
//!
//! The previous rmake.rs iteration of this test was flaky for unknown reason on
//! `i686-pc-windows-gnu` *specifically*, so assertion failures in this test was made extremely
//! verbose to help diagnose why the ICE messages was different. It appears that backtraces on
//! `i686-pc-windows-gnu` specifically are quite unpredictable in how many backtrace frames are
//! involved.

//@ ignore-cross-compile (exercising ICE dump on host)
//@ ignore-i686-pc-windows-gnu (unwind mechanism produces unpredictable backtraces)

use std::cell::OnceCell;
use std::path::{Path, PathBuf};

use run_make_support::{
    cwd, filename_contains, has_extension, has_prefix, rfs, run_in_tmpdir, rustc,
    shallow_find_files,
};

#[derive(Debug)]
struct IceDump {
    name: &'static str,
    path: PathBuf,
    message: String,
}

impl IceDump {
    fn lines_count(&self) -> usize {
        self.message.lines().count()
    }
}

#[track_caller]
fn assert_ice_len_equals(left: &IceDump, right: &IceDump) {
    let left_len = left.lines_count();
    let right_len = right.lines_count();

    if left_len != right_len {
        eprintln!("=== {} ICE MESSAGE ({} lines) ====", left.name, left_len);
        eprintln!("{}", left.message);

        eprintln!("=== {} ICE MESSAGE ({} lines) ====", right.name, right_len);
        eprintln!("{}", right.message);

        eprintln!("====================================");
        panic!(
            "ICE message length mismatch: {} has {} lines but {} has {} lines",
            left.name, left_len, right.name, right_len
        );
    }
}

fn find_ice_dumps_in_dir<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    shallow_find_files(dir, |path| has_prefix(path, "rustc-ice") && has_extension(path, "txt"))
}

// Assert only one `rustc-ice*.txt` ICE file exists, and extract the ICE message from the ICE file.
#[track_caller]
fn extract_exactly_one_ice_file<P: AsRef<Path>>(name: &'static str, dir: P) -> IceDump {
    let ice_files = find_ice_dumps_in_dir(dir);
    assert_eq!(ice_files.len(), 1); // There should only be 1 ICE file.
    let path = ice_files.get(0).unwrap();
    let message = rfs::read_to_string(path);
    IceDump { name, path: path.to_path_buf(), message }
}

fn main() {
    // Establish baseline ICE message.
    let default_ice_dump = OnceCell::new();
    run_in_tmpdir(|| {
        rustc().env("RUSTC_ICE", cwd()).input("lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
        let dump = extract_exactly_one_ice_file("baseline", cwd());
        // Ensure that the ICE dump path doesn't contain `:`, because they cause problems on
        // Windows.
        assert!(!filename_contains(&dump.path, ":"), "{} contains `:`", dump.path.display());
        // Some of the expected strings in an ICE file should appear.
        assert!(dump.message.contains("thread 'rustc' panicked at"));
        assert!(dump.message.contains("stack backtrace:"));
        default_ice_dump.set(dump).unwrap();
    });
    let default_ice_dump = default_ice_dump.get().unwrap();

    test_backtrace_short(default_ice_dump);
    test_backtrace_full(default_ice_dump);
    test_backtrace_disabled(default_ice_dump);
    test_ice_dump_disabled();

    test_metrics_dir(default_ice_dump);
}

#[track_caller]
fn test_backtrace_short(baseline: &IceDump) {
    run_in_tmpdir(|| {
        rustc()
            .env("RUSTC_ICE", cwd())
            .input("lib.rs")
            .env("RUST_BACKTRACE", "short")
            .arg("-Ztreat-err-as-bug=1")
            .run_fail();
        let dump = extract_exactly_one_ice_file("RUST_BACKTRACE=short", cwd());
        // Backtrace length in dump shouldn't be changed by `RUST_BACKTRACE`.
        assert_ice_len_equals(baseline, &dump);
    });
}

#[track_caller]
fn test_backtrace_full(baseline: &IceDump) {
    run_in_tmpdir(|| {
        rustc()
            .env("RUSTC_ICE", cwd())
            .input("lib.rs")
            .env("RUST_BACKTRACE", "full")
            .arg("-Ztreat-err-as-bug=1")
            .run_fail();
        let dump = extract_exactly_one_ice_file("RUST_BACKTRACE=full", cwd());
        // Backtrace length in dump shouldn't be changed by `RUST_BACKTRACE`.
        assert_ice_len_equals(baseline, &dump);
    });
}

#[track_caller]
fn test_backtrace_disabled(baseline: &IceDump) {
    run_in_tmpdir(|| {
        rustc()
            .env("RUSTC_ICE", cwd())
            .input("lib.rs")
            .env("RUST_BACKTRACE", "0")
            .arg("-Ztreat-err-as-bug=1")
            .run_fail();
        let dump = extract_exactly_one_ice_file("RUST_BACKTRACE=disabled", cwd());
        // Backtrace length in dump shouldn't be changed by `RUST_BACKTRACE`.
        assert_ice_len_equals(baseline, &dump);
    });
}

#[track_caller]
fn test_ice_dump_disabled() {
    // The ICE dump is explicitly disabled. Therefore, this should produce no files.
    run_in_tmpdir(|| {
        rustc().env("RUSTC_ICE", "0").input("lib.rs").arg("-Ztreat-err-as-bug=1").run_fail();
        let ice_files = find_ice_dumps_in_dir(cwd());
        assert!(ice_files.is_empty(), "there should be no ICE files if `RUSTC_ICE=0` is set");
    });
}

#[track_caller]
fn test_metrics_dir(baseline: &IceDump) {
    test_flag_only(baseline);
    test_flag_and_env(baseline);
}

#[track_caller]
fn test_flag_only(baseline: &IceDump) {
    run_in_tmpdir(|| {
        let metrics_arg = format!("-Zmetrics-dir={}", cwd().display());
        rustc()
            .env_remove("RUSTC_ICE") // prevent interference from environment
            .input("lib.rs")
            .arg("-Ztreat-err-as-bug=1")
            .arg(metrics_arg)
            .run_fail();
        let dump = extract_exactly_one_ice_file("-Zmetrics-dir only", cwd());
        assert_ice_len_equals(baseline, &dump);
    });
}

#[track_caller]
fn test_flag_and_env(baseline: &IceDump) {
    run_in_tmpdir(|| {
        let metrics_arg = format!("-Zmetrics-dir={}", cwd().display());
        let real_dir = cwd().join("actually_put_ice_here");
        rfs::create_dir(&real_dir);
        rustc()
            .input("lib.rs")
            .env("RUSTC_ICE", &real_dir)
            .arg("-Ztreat-err-as-bug=1")
            .arg(metrics_arg)
            .run_fail();

        let cwd_ice_files = find_ice_dumps_in_dir(cwd());
        assert!(cwd_ice_files.is_empty(), "RUSTC_ICE should override -Zmetrics-dir");

        let dump = extract_exactly_one_ice_file("RUSTC_ICE overrides -Zmetrics-dir", real_dir);
        assert_ice_len_equals(baseline, &dump);
    });
}
