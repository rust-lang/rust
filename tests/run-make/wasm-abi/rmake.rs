//@ only-wasm32-wasip1
//@ needs-wasmtime

use run_make_support::{cmd, rustc};
use std::path::Path;

fn main() {
    rustc().input("foo.rs").target("wasm32-wasip1").run();

    let file = Path::new("foo.wasm");

    run(&file, "return_two_i32", "1\n2\n");
    run(&file, "return_two_i64", "3\n4\n");
    run(&file, "return_two_f32", "5\n6\n");
    run(&file, "return_two_f64", "7\n8\n");
    run(&file, "return_mishmash", "9\n10\n11\n12\n13\n14\n");
    run(&file, "call_imports", "");
}

fn run(file: &Path, method: &str, expected_output: &str) {
    cmd("wasmtime")
        .arg("run")
        .arg("--preload=host=host.wat")
        .arg("--invoke")
        .arg(method)
        .arg(file)
        .run()
        .assert_stdout_equals(expected_output);
}
