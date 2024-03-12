extern crate run_make_support;

use run_make_support::{out_dir, rustc};
use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("TARGET").unwrap() != "wasm32-wasip1" {
        return;
    }

    rustc().arg("foo.rs").arg("--target=wasm32-wasip1").run();
    let file = out_dir().join("foo.wasm");

    let has_wasmtime = match Command::new("wasmtime").arg("--version").output() {
        Ok(s) => s.status.success(),
        _ => false,
    };
    if !has_wasmtime {
        println!("skipping test, wasmtime isn't available");
        return;
    }

    run(&file, "return_two_i32", "1\n2\n");
    run(&file, "return_two_i64", "3\n4\n");
    run(&file, "return_two_f32", "5\n6\n");
    run(&file, "return_two_f64", "7\n8\n");
    run(&file, "return_mishmash", "9\n10\n11\n12\n13\n14\n");
    run(&file, "call_imports", "");
}

fn run(file: &Path, method: &str, expected_output: &str) {
    let output = Command::new("wasmtime")
        .arg("run")
        .arg("--preload=host=host.wat")
        .arg("--invoke")
        .arg(method)
        .arg(file)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(expected_output, String::from_utf8_lossy(&output.stdout));
}
