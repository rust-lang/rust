// Check how `native` interacts with targets that require consistent
// `-Ctarget-cpu` values across crates.
//
// First, the test derives a custom target from the host target and removes
// `requires-consistent-cpu` if present. Since host and target architecture match,
// this target prints `native` in `--print target-cpus`.
// Then, the test derives a second custom target from the host target and sets
// `requires-consistent-cpu` to true. This target must not print `native` in
// `--print target-cpus`.
// Finally, the test verifies that `-Ctarget-cpu=native` is rejected when using
// the second custom target.

use std::fs;

use run_make_support::*;
use serde_json::{Value, json};

fn main() {
    let is_native_cpu_line = |line: &str| line.trim_start().starts_with("native ");
    let target_cpus = |target: &str| {
        rustc().target(target).arg("-Zunstable-options").print("target-cpus").run().stdout_utf8()
    };

    let host = rustc().print("host-tuple").run().stdout_utf8().trim().to_owned();

    let host_target_with_cpu_mismatch_allowed = custom_host(&host, false);
    let cpus = target_cpus(&host_target_with_cpu_mismatch_allowed);
    assert!(
        cpus.lines().any(is_native_cpu_line),
        "`native` should be printed for the host target without `requires-consistent-cpu`;\n\
         output was:\n{cpus}"
    );

    let host_target_with_requires_consistent_cpu = custom_host(&host, true);
    let cpus = target_cpus(&host_target_with_requires_consistent_cpu);
    assert!(
        !cpus.lines().any(is_native_cpu_line),
        "`native` must not be printed for targets with `requires-consistent-cpu`;\n\
         output was:\n{cpus}"
    );

    rustc()
        .arg("-Zunstable-options")
        .target(&host_target_with_requires_consistent_cpu)
        .target_cpu("native")
        .input("empty.rs")
        .run_fail()
        .assert_stderr_contains("`-Ctarget-cpu=native` is not allowed")
        .assert_stderr_contains("requires consistent `-Ctarget-cpu` values");
}

fn custom_host(host: &str, requires_consistent_cpu: bool) -> String {
    let json = rustc()
        .arg("-Zunstable-options")
        .target(host)
        .print("target-spec-json")
        .run()
        .stdout_utf8();

    let mut spec: Value = serde_json::from_str(&json).unwrap();
    let spec = spec.as_object_mut().expect("expected target-spec JSON to be an object");
    let filename = if requires_consistent_cpu {
        spec.insert("requires-consistent-cpu".to_string(), json!(true));
        format!("{host}-requires-consistent-cpu.json")
    } else {
        spec.remove("requires-consistent-cpu");
        format!("{host}-cpu-mismatch-allowed.json")
    };

    fs::write(&filename, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    filename
}
