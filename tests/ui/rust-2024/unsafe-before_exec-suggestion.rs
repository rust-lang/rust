//@ edition:2015
//@ only-unix
//@ run-rustfix

#![deny(deprecated_safe_2024)]

use std::process::Command;
use std::os::unix::process::CommandExt;

#[allow(deprecated)]
fn main() {
    let mut cmd = Command::new("sleep");
    cmd.before_exec(|| Ok(()));
    //~^ ERROR call to deprecated safe function
    //~| WARN this is accepted in the current edition
    drop(cmd);
}
