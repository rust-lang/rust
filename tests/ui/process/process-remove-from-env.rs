//@ run-pass
//@ needs-subprocess
//@ ignore-vxworks no 'env'
//@ ignore-fuchsia no 'env'
//@ ignore-ios no 'env'
//@ ignore-tvos no 'env'
//@ ignore-watchos no 'env'
//@ ignore-visionos no 'env'

use std::process::Command;
use std::env;

#[cfg(all(unix, not(target_os="android")))]
pub fn env_cmd() -> Command {
    Command::new("env")
}
#[cfg(target_os="android")]
pub fn env_cmd() -> Command {
    let mut cmd = Command::new("/system/bin/sh");
    cmd.arg("-c").arg("set");
    cmd
}

#[cfg(windows)]
pub fn env_cmd() -> Command {
    let mut cmd = Command::new("cmd");
    cmd.arg("/c").arg("set");
    cmd
}

fn main() {
    // save original environment
    let old_env = env::var_os("RUN_TEST_NEW_ENV");

    env::set_var("RUN_TEST_NEW_ENV", "123");

    let mut cmd = env_cmd();
    cmd.env_remove("RUN_TEST_NEW_ENV");

    // restore original environment
    match old_env {
        None => env::remove_var("RUN_TEST_NEW_ENV"),
        Some(val) => env::set_var("RUN_TEST_NEW_ENV", &val)
    }

    let result = cmd.output().unwrap();
    let output = String::from_utf8_lossy(&result.stdout);

    assert!(!output.contains("RUN_TEST_NEW_ENV"),
            "found RUN_TEST_NEW_ENV inside of:\n\n{}", output);
}
