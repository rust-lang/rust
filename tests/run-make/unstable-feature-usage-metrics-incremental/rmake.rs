//! This test checks if unstable feature usage metric dump files `unstable-feature-usage*.json` work
//! as expected.
//!
//! - Basic sanity checks on a default ICE dump.
//!
//! See <https://github.com/rust-lang/rust/issues/129485>.
//!
//! # Test history
//!
//! - forked from dump-ice-to-disk test, which has flakeyness issues on i686-mingw, I'm assuming
//! those will be present in this test as well on the same platform

//@ ignore-windows
//FIXME(#128911): still flakey on i686-mingw.

use std::path::{Path, PathBuf};

use run_make_support::rfs::create_dir_all;
use run_make_support::{
    cwd, filename_contains, has_extension, rfs, run_in_tmpdir, rustc, serde_json,
    shallow_find_files,
};

fn find_feature_usage_metrics<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    shallow_find_files(dir, |path| {
        if filename_contains(path, "unstable_feature_usage") && has_extension(path, "json") {
            true
        } else {
            dbg!(path);
            false
        }
    })
}

fn main() {
    test_metrics_dump();
    test_metrics_errors();
}

#[track_caller]
fn test_metrics_dump() {
    run_in_tmpdir(|| {
        let metrics_dir = cwd().join("metrics");
        create_dir_all(&metrics_dir);
        rustc()
            .input("main.rs")
            .incremental("incremental")
            .env("RUST_BACKTRACE", "short")
            .arg(format!("-Zmetrics-dir={}", metrics_dir.display()))
            .run();
        let mut metrics = find_feature_usage_metrics(&metrics_dir);
        let json_path =
            metrics.pop().expect("there should be one metrics file in the output directory");

        // After the `pop` above, there should be no files left.
        assert!(
            metrics.is_empty(),
            "there should be no more than one metrics file in the output directory"
        );

        let message = rfs::read_to_string(json_path);
        let mut parsed: serde_json::Value =
            serde_json::from_str(&message).expect("metrics should be dumped as json");
        // remove timestamps
        assert!(parsed["lib_features"][0]["timestamp"].is_number());
        assert!(parsed["lang_features"][0]["timestamp"].is_number());
        parsed["lib_features"][0]["timestamp"] = serde_json::json!(null);
        parsed["lang_features"][0]["timestamp"] = serde_json::json!(null);
        let expected = serde_json::json!(
            {
                "lib_features":[{"symbol":"ascii_char", "timestamp":null}],
                "lang_features":[{"symbol":"box_patterns","since":null, "timestamp":null}]
            }
        );

        assert_eq!(expected, parsed);
    });
}

#[track_caller]
fn test_metrics_errors() {
    run_in_tmpdir(|| {
        rustc()
            .input("main.rs")
            .incremental("incremental")
            .env("RUST_BACKTRACE", "short")
            .arg("-Zmetrics-dir=invaliddirectorythatdefinitelydoesntexist")
            .run_fail()
            .assert_stderr_contains(
                "error: cannot dump feature usage metrics: No such file or directory",
            )
            .assert_stdout_not_contains("internal compiler error");
    });
}
