extern crate compiletest_rs as compiletest;

use std::path::PathBuf;

fn run_mode(mode: &'static str) {
    let mut config = compiletest::default_config();
    config.rustc_path = "target/debug/miri".into();
    let path = std::env::var("RUST_SYSROOT").expect("env variable `RUST_SYSROOT` not set");
    config.target_rustcflags = Some(format!("--sysroot {}", path));
    config.host_rustcflags = Some(format!("--sysroot {}", path));
    let cfg_mode = mode.parse().ok().expect("Invalid mode");

    config.mode = cfg_mode;
    config.src_base = PathBuf::from(format!("tests/{}", mode));

    compiletest::run_tests(&config);
}

#[test]
fn compile_test() {
    run_mode("compile-fail");
    run_mode("run-pass");
}
