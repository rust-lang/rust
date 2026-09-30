// Verifies that when a crate hash is required solely for a metrics dir, that
// crate hash is successfully generated, either by the fallback mechanism or
// metadata generation.
// See https://github.com/rust-lang/rust/issues/163426.

//@ ignore-cross-compile

use std::path::{Path, PathBuf};

use run_make_support::rfs::create_dir_all;
use run_make_support::{
    cwd, filename_contains, has_extension, run_in_tmpdir, rustc, rustdoc, shallow_find_files,
};

fn find_feature_usage_metrics<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    shallow_find_files(dir, |path| {
        filename_contains(path, "unstable_feature_usage") && has_extension(path, "json")
    })
}

fn main() {
    let metrics_dir = "metrics-rustdoc";
    create_dir_all(&metrics_dir);
    rustdoc()
        .input("lib.rs")
        .out_dir("doc")
        .env("RUST_BACKTRACE", "short")
        .arg(format!("-Zmetrics-dir={}", metrics_dir))
        .run()
        .assert_stderr_not_contains("internal compiler error");
    assert_eq!(
        find_feature_usage_metrics(&metrics_dir).len(),
        1,
        "rustdoc should dump exactly one metrics file"
    );

    let metrics_dir = "metrics-rustc";
    create_dir_all(&metrics_dir);
    rustc()
        .input("main.rs")
        .crate_type("bin")
        .env("RUST_BACKTRACE", "short")
        .arg(format!("-Zmetrics-dir={}", metrics_dir))
        .run()
        .assert_stderr_not_contains("internal compiler error");
    assert_eq!(
        find_feature_usage_metrics(&metrics_dir).len(),
        1,
        "rustc should dump exactly one metrics file"
    );
}
