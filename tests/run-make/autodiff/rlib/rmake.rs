//@ needs-enzyme
//@ ignore-cross-compile

use run_make_support::{cwd, run, rustc};

fn main() {
    // Build the dependency crate.
    rustc()
        .input("dep.rs")
        .arg("-Zautodiff=Enable")
        .arg("--edition=2024")
        .arg("-Copt-level=3")
        .arg("--crate-name=simple_dep")
        .arg("-Clinker-plugin-lto")
        .arg("--crate-type=lib")
        .emit("dep-info,metadata,link")
        .run();

    let cwd = cwd();
    let cwd_str = cwd.to_string_lossy();

    let mydep = format!("-Ldependency={cwd_str}");

    let simple_dep_rlib =
        format!("--extern=simple_dep={}", cwd.join("libsimple_dep.rlib").to_string_lossy());

    // Build the main library that depends on `simple_dep`.
    rustc()
        .input("lib.rs")
        .arg("-Zautodiff=Enable")
        .arg("--edition=2024")
        .arg("-Copt-level=3")
        .arg("--crate-name=foo")
        .arg("-Clinker-plugin-lto")
        .arg("--crate-type=lib")
        .emit("dep-info,metadata,link")
        .arg(&mydep)
        .arg(&simple_dep_rlib)
        .run();

    let foo_rlib = format!("--extern=foo={}", cwd.join("libfoo.rlib").to_string_lossy());

    // Build the final binary linking both rlibs.
    rustc()
        .input("main.rs")
        .arg("-Zautodiff=Enable")
        .arg("--edition=2024")
        .arg("-Copt-level=3")
        .arg("--crate-name=foo")
        .arg("-Clto=fat")
        .arg("--crate-type=bin")
        .emit("dep-info,link")
        .arg(&mydep)
        .arg(&foo_rlib)
        .arg(&simple_dep_rlib)
        .run();

    // Run the binary and check its output.
    let binary = run("foo");
    assert!(binary.status().success(), "binary failed to run");

    let binary_out = binary.stdout();
    let output = String::from_utf8_lossy(&binary_out);
    assert!(output.contains("output1: 4.488727439718245"));
    assert!(output.contains("output2: 3.3108023673168265"));
}
