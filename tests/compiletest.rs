extern crate compiletest_rs as compiletest;

use std::path::PathBuf;

fn run_mode(mode: &'static str) {
    // FIXME: read directories in sysroot/lib/rustlib and generate the test targets from that
    let targets = &["x86_64-unknown-linux-gnu", "i686-unknown-linux-gnu"];

    for &target in targets {
        let mut config = compiletest::default_config();
        config.rustc_path = "target/debug/miri".into();
        let path = std::env::var("RUST_SYSROOT").expect("env variable `RUST_SYSROOT` not set");
        config.run_lib_path = format!("{}/lib/rustlib/{}/lib", path, target);
        let path = format!("--sysroot {}", path);
        config.target_rustcflags = Some(path.clone());
        config.host_rustcflags = Some(path);
        let cfg_mode = mode.parse().ok().expect("Invalid mode");

        config.mode = cfg_mode;
        config.src_base = PathBuf::from(format!("tests/{}", mode));
        config.target = target.to_owned();
        compiletest::run_tests(&config);
    }
}

#[test]
fn compile_test() {
    run_mode("compile-fail");
    run_mode("run-pass");
    run_mode("run-fail");
}
