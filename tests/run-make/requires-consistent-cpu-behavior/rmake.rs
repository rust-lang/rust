// Check that cross-crate `-Ctarget-cpu` mismatches are accepted when
// `requires-consistent-cpu` is false, and rejected when it is true.
//
// The test first compiles two crates for a built-in target using different
// `-Ctarget-cpu` values. It asserts that the target's default spec either
// omits `requires-consistent-cpu` or sets it to false, so this compilation
// must succeed.
//
// It then repeats the same compilation using a copy of that target spec with
// `requires-consistent-cpu` set to true. This time the compilation must fail.
use std::fs;

use run_make_support::*;
use serde_json::{Value, json, to_string};

fn main() {
    let target = "x86_64-unknown-linux-gnu";

    let json = rustc()
        .arg("-Zunstable-options")
        .target(target)
        .print("target-spec-json")
        .run()
        .stdout_utf8();

    let mut spec: Value = serde_json::from_str(&json).unwrap();
    match spec.get("requires-consistent-cpu") {
        None | Some(Value::Bool(false)) => {}
        other => panic!(
            "expected `requires-consistent-cpu` to be false or absent for {target}, got {other:?}"
        ),
    }
    // With `requires-consistent-cpu = false`, mismatched `-Ctarget-cpu`
    // values across crates are accepted.
    compile(target, false);

    // Adapt the target-spec-json and write to file
    spec.as_object_mut()
        .expect("Expected the target-spec-json to be an object")
        .insert("requires-consistent-cpu".to_string(), json!(true));
    let filename = format!("{}-consistent-cpu.json", target);
    fs::write(&filename, to_string(&spec).unwrap()).unwrap();

    // With `requires-consistent-cpu = true`, the same mismatch is rejected.
    compile(&filename, true);
}

fn compile(target: &str, must_fail: bool) {
    // compile `dependency.rs`
    rustc()
        .arg("-Zunstable-options")
        .target(target)
        .target_cpu("x86-64")
        .input("dependency.rs")
        .run();

    // compile `main.rs` with different cpu
    let mut invocation = rustc();
    invocation.arg("-Zunstable-options").target(target).target_cpu("x86-64-v2").input("main.rs");

    if must_fail {
        invocation
            .run_fail()
            .assert_stderr_contains("mixing `-Ctarget-cpu` will cause an ABI mismatch");
    } else {
        invocation.run();
    }
}
