//! Test if compilation with has-thread-local=false works, and if the output
//! has indeed no fast TLS variables.

//@ only-apple

use run_make_support::serde_json::{self, Value};
use run_make_support::{cargo, llvm_nm, rfs, rustc};

fn main() {
    let output =
        rustc().print("target-spec-json").args(["-Z", "unstable-options"]).run().stdout_utf8();

    let mut target_json: Value = serde_json::from_str(&output).unwrap();
    let has_thread_local = &mut target_json["has-thread-local"];
    assert!(matches!(has_thread_local, Value::Bool(true)), "{:?}", has_thread_local);
    *has_thread_local = Value::Bool(false);

    let out_path = "t.json";
    rfs::write(out_path, serde_json::to_string(&target_json).unwrap());

    cargo()
        .args([
            "b",
            "--manifest-path",
            "tls_test/Cargo.toml",
            "--target",
            "t.json",
            "-Zbuild-std=std,core,panic_abort",
        ])
        .run();

    // If a binary has any fast TLS variables, it should also contain the symbols
    // __tlv_bootstrap and __tlv_atexit. We don't want them.
    let output = llvm_nm().arg("tls_test/target/t/debug/tls_test").run().stdout_utf8();
    assert!(!output.contains("_tlv_bootstrap"));
    assert!(!output.contains("_tlv_atexit"));
}
