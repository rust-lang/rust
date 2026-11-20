//@ run-pass

//@ compile-flags:-Cstrip=none
//@ compile-flags:-g -Csplit-debuginfo=unpacked
//@ only-apple
//@ ignore-remote needs the compiler-produced `.o` file to be copied to the device

use std::process::Command;
use std::str;

#[inline(never)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        println!("{}", std::backtrace::Backtrace::force_capture());
        return;
    }
    let out = Command::new(&args[0]).env("RUST_BACKTRACE", "1").arg("foo").output().unwrap();
    let output = format!(
        "{}\n{}",
        str::from_utf8(&out.stdout).unwrap(),
        str::from_utf8(&out.stderr).unwrap(),
    );
    if out.status.success() && output.contains(file!()) {
        return;
    }
    println!("status: {}", out.status);
    println!("child output:\n\t{}", output.replace("\n", "\n\t"));
    panic!("failed to find {:?} in output", file!());
}
