// This test ignores some platforms as the particular extension trait used
// to demonstrate the issue is only available on unix. This is fine as
// the fix to suggested paths is not platform-dependent and will apply on
// these platforms also.

// ignore-windows
// ignore-emscripten
// ignore-sgx no processes

use std::process::Command;
// use std::os::unix::process::CommandExt;

fn main() {
    Command::new("echo").arg("hello").exec();
//~^ ERROR no method named `exec`
}
