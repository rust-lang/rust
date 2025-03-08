//! Test if compilation with has-thread-local=false works, and if the output
//! has indeed no fast TLS variables.

//@ only-apple

use run_make_support::run::cmd;
use run_make_support::{cargo, rfs, rustc};

fn main() {
    let output =
        rustc().print("target-spec-json").args(["-Z", "unstable-options"]).run().stdout_utf8();
    let to_search = r#""has-thread-local": true"#;
    assert!(output.contains(to_search));
    let output = output.replace(to_search, r#""has-thread-local": false"#);

    let out_path = "t.json";
    rfs::write(out_path, output);

    cargo()
        .current_dir("tls_test")
        .args(["b", "--target", "../t.json", "-Zbuild-std=std,core,panic_abort"])
        .run();

    let output = cmd("nm").arg("tls_test/target/t/debug/tls_test").run().stdout_utf8();
    // If a binary has any fast TLS variables, it should also contain the symbols
    // __tlv_bootstrap and __tlv_atexit. We don't want them.
    assert!(!output.contains("__tlv_bootstrap"));
}
