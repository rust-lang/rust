// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::Command;
use std::os;

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
    let old_env = os::getenv("RUN_TEST_NEW_ENV");

    os::setenv("RUN_TEST_NEW_ENV", "123");

    let mut cmd = env_cmd();
    cmd.env_remove("RUN_TEST_NEW_ENV");

    // restore original environment
    match old_env {
        None => os::unsetenv("RUN_TEST_NEW_ENV"),
        Some(val) => os::setenv("RUN_TEST_NEW_ENV", val.as_slice())
    }

    let prog = cmd.spawn().unwrap();
    let result = prog.wait_with_output().unwrap();
    let output = String::from_utf8_lossy(result.output.as_slice());

    assert!(!output.as_slice().contains("RUN_TEST_NEW_ENV"),
            "found RUN_TEST_NEW_ENV inside of:\n\n{}", output);
}
