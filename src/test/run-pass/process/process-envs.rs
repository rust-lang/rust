// run-pass
// ignore-cloudabi no processes
// ignore-emscripten no processes
// ignore-sgx no processes

use std::process::Command;
use std::env;
use std::collections::HashMap;

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

    // create filtered environment vector
    let filtered_env : HashMap<String, String> =
        env::vars().filter(|&(ref k, _)| k == "PATH").collect();

    let mut cmd = env_cmd();
    cmd.env_clear();
    cmd.envs(&filtered_env);

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
