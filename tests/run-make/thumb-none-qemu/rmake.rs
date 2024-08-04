//! How to run this
//! $ ./x.py clean
//! $ ./x.py test --target thumbv6m-none-eabi,thumbv7m-none-eabi tests/run-make
//!
//! For supported targets, see `example/.cargo/config`

//@ only-thumb

use std::path::PathBuf;

use run_make_support::{cmd, env_var};

const CRATE: &str = "example";

fn main() {
    std::env::set_current_dir(CRATE).unwrap();

    let target = env_var("TARGET");
    let bootstrap_cargo = env_var("BOOTSTRAP_CARGO");
    let path = env_var("PATH");
    let rustc = env_var("RUSTC");

    let target_dir = PathBuf::from("target");
    let manifest_path = PathBuf::from("Cargo.toml");

    let debug = {
        let mut cmd = cmd(&bootstrap_cargo);
        cmd.args(&["run", "--target", &target])
            .env("RUSTFLAGS", "-C linker=arm-none-eabi-ld -C link-arg=-Tlink.x")
            .env("CARGO_TARGET_DIR", &target_dir)
            .env("PATH", &path)
            .env("RUSTC", &rustc);
        cmd.run()
    };

    let stdout = debug.stdout_utf8();
    assert!(stdout.contains("x = 42"), "stdout: {:?}", stdout);

    let release = {
        let mut cmd = cmd(&bootstrap_cargo);
        cmd.args(&["run", "--release", "--target", &target])
            .env("RUSTFLAGS", "-C linker=arm-none-eabi-ld -C link-arg=-Tlink.x")
            .env("CARGO_TARGET_DIR", &target_dir)
            .env("PATH", &path)
            .env("RUSTC", &rustc);
        cmd.run()
    };

    let stdout = release.stdout_utf8();
    assert!(stdout.contains("x = 42"), "stdout: {:?}", stdout);
}
