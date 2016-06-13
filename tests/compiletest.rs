extern crate compiletest_rs as compiletest;

use std::path::PathBuf;

fn run_mode(mode: &'static str) {
    // Disable rustc's new error fomatting. It breaks these tests.
    std::env::remove_var("RUST_NEW_ERROR_FORMAT");

    // Taken from https://github.com/Manishearth/rust-clippy/pull/911.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    let sysroot = match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ => option_env!("RUST_SYSROOT")
            .expect("need to specify RUST_SYSROOT env var or use rustup or multirust")
            .to_owned(),
    };
    let sysroot_flag = format!("--sysroot {} -Dwarnings", sysroot);

    // FIXME: read directories in sysroot/lib/rustlib and generate the test targets from that
    let targets = &["x86_64-unknown-linux-gnu", "i686-unknown-linux-gnu"];

    for &target in targets {
        let mut config = compiletest::default_config();
        config.host_rustcflags = Some(sysroot_flag.clone());
        config.mode = mode.parse().expect("Invalid mode");
        config.run_lib_path = format!("{}/lib/rustlib/{}/lib", sysroot, target);
        config.rustc_path = "target/debug/miri".into();
        config.src_base = PathBuf::from(format!("tests/{}", mode));
        config.target = target.to_owned();
        config.target_rustcflags = Some(sysroot_flag.clone());
        compiletest::run_tests(&config);
    }
}

#[test]
fn compile_test() {
    run_mode("compile-fail");
    run_mode("run-pass");
}
