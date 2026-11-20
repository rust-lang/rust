#![feature(exit_status_error)]
#![warn(rust_2018_idioms, unused_lifetimes)]

mod test_utils;

use std::fs;
use std::path::Path;
use std::process::Command;
use test_utils::IS_RUSTC_TEST_SUITE;

#[test]
fn driver() {
    if IS_RUSTC_TEST_SUITE {
        return;
    }

    let cargo = env!("CARGO");
    let driver = env!("CARGO_BIN_EXE_clippy-driver");
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let tmp_arg_file = tmp.join("clippy_driver_arg_file.txt").into_string().unwrap();
    let arg_file_arg = String::from_iter(["@", &*tmp_arg_file]);

    fs::create_dir_all(tmp).unwrap();
    fs::write(&tmp_arg_file, "--sysroot=sysroot\n".as_bytes()).unwrap();

    // Check that the sysroot is set correctly.
    assert_eq!(
        run_output(Command::new(driver).args(["--print", "sysroot"])),
        // Run via. cargo so we get the same rustc version that built clippy.
        run_output(Command::new(cargo).args(["rustc", "-Z", "unstable-options", "--print", "sysroot"])),
    );
    assert_eq!(
        run_output(Command::new(driver).args(["--sysroot", "sysroot", "--print", "sysroot"])),
        "sysroot\n",
    );
    assert_eq!(
        run_output(Command::new(driver).args([&*arg_file_arg, "--print", "sysroot"])),
        "sysroot\n",
    );
    assert_eq!(
        run_output(
            Command::new(driver)
                .env("SYSROOT", "sysroot")
                .args(["--print", "sysroot"])
        ),
        "sysroot\n",
    );

    // Check that sysroot is set only once.
    assert_eq!(
        run_output(Command::new(driver).env("SYSROOT", "sysroot_env").args([
            "--sysroot",
            "sysroot",
            "--print",
            "sysroot"
        ])),
        "sysroot\n",
    );
    assert_eq!(
        run_output(
            Command::new(driver)
                .env("SYSROOT", "sysroot_env")
                .args([&*arg_file_arg, "--print", "sysroot"])
        ),
        "sysroot\n",
    );

    // Check version.
    assert_ne!(
        run_output(Command::new(driver).args(["-V"])).strip_prefix("clippy "),
        None
    );

    // Check rustc forwarding.
    assert_ne!(
        run_output(Command::new(driver).args(["--rustc", "-V"])).strip_prefix("rustc "),
        None
    );
}

fn run_output(cmd: &mut Command) -> String {
    let output = cmd.output().unwrap();
    output.status.exit_ok().unwrap();
    String::from_utf8(output.stdout).unwrap()
}
