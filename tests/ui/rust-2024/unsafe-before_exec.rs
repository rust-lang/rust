//@ revisions: e2021 e2024
//@ only-unix
//@[e2021] edition: 2021
//@[e2021] check-pass
//@[e2024] edition: 2024

use std::process::Command;
use std::os::unix::process::CommandExt;

#[allow(deprecated)]
fn main() {
    let mut cmd = Command::new("sleep");
    cmd.before_exec(|| Ok(()));
    //[e2024]~^ ERROR call to unsafe function `before_exec` is unsafe
    drop(cmd); // we don't actually run the command.
}
