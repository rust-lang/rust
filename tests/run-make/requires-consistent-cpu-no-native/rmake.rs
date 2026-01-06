// Check how `native` interacts with targets that require consistent
// `-Ctarget-cpu` values across crates.
//
// The test derives a custom target from the host target and sets
// `requires-consistent-cpu` to true. Since host and target architecture match,
// the unmodified host target normally prints `native` in `--print target-cpus`.
// The custom target must not print it, and using `-Ctarget-cpu=native` with
// that custom target must be rejected.

use std::fs;

use run_make_support::*;
use serde_json::{Value, json};

fn main() {
    let host = rustc().print("host-tuple").run().stdout_utf8().trim().to_owned();

    let host_cpus = target_cpus(&host);
    assert!(
        host_cpus.lines().any(is_native_cpu_line),
        "`native` should be printed for the unmodified host target;\n\
         output was:\n{host_cpus}"
    );

    let custom_target = custom_host_target_with_requires_consistent_cpu(&host);

    let custom_cpus = target_cpus(&custom_target);
    assert!(
        !custom_cpus.lines().any(is_native_cpu_line),
        "`native` must not be printed for targets with `requires-consistent-cpu`;\n\
         output was:\n{custom_cpus}"
    );

    rustc()
        .arg("-Zunstable-options")
        .target(&custom_target)
        .target_cpu("native")
        .input("empty.rs")
        .run_fail()
        .assert_stderr_contains("`-Ctarget-cpu=native` is not allowed")
        .assert_stderr_contains("requires consistent `-Ctarget-cpu` values");
}

fn target_cpus(target: &str) -> String {
    rustc().arg("-Zunstable-options").target(target).print("target-cpus").run().stdout_utf8()
}

fn custom_host_target_with_requires_consistent_cpu(host: &str) -> String {
    let json = rustc()
        .arg("-Zunstable-options")
        .target(host)
        .print("target-spec-json")
        .run()
        .stdout_utf8();

    let mut spec: Value = serde_json::from_str(&json).unwrap();

    spec.as_object_mut()
        .expect("expected target-spec JSON to be an object")
        .insert("requires-consistent-cpu".to_string(), json!(true));

    let filename = format!("{host}-requires-consistent-cpu.json");
    fs::write(&filename, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    filename
}

fn is_native_cpu_line(line: &str) -> bool {
    line.trim_start().starts_with("native ")
}
