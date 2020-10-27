// run-pass
// ignore-android FIXME #17520
// ignore-emscripten spawning processes is not supported
// ignore-openbsd no support for libbacktrace without filename
// ignore-sgx no processes
// ignore-msvc see #62897 and `backtrace-debuginfo.rs` test
// compile-flags:-g

#![feature(backtrace)]

use std::env;
use std::process::Command;
use std::str;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "force" {
        println!("stack backtrace:\n{}", std::backtrace::Backtrace::force_capture());
    } else if args.len() >= 2 {
        println!("stack backtrace:\n{}", std::backtrace::Backtrace::capture());
    } else {
        runtest(&args[0]);
        println!("test ok");
    }
}

fn runtest(me: &str) {
    env::remove_var("RUST_BACKTRACE");
    env::remove_var("RUST_LIB_BACKTRACE");

    let p = Command::new(me).arg("a").env("RUST_BACKTRACE", "1").output().unwrap();
    assert!(p.status.success());
    assert!(String::from_utf8_lossy(&p.stdout).contains("stack backtrace:\n"));
    assert!(String::from_utf8_lossy(&p.stdout).contains("backtrace::main"));

    let p = Command::new(me).arg("a").env("RUST_BACKTRACE", "0").output().unwrap();
    assert!(p.status.success());
    assert!(String::from_utf8_lossy(&p.stdout).contains("disabled backtrace\n"));

    let p = Command::new(me).arg("a").output().unwrap();
    assert!(p.status.success());
    assert!(String::from_utf8_lossy(&p.stdout).contains("disabled backtrace\n"));

    let p = Command::new(me)
        .arg("a")
        .env("RUST_LIB_BACKTRACE", "1")
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap();
    assert!(p.status.success());
    assert!(String::from_utf8_lossy(&p.stdout).contains("stack backtrace:\n"));

    let p = Command::new(me)
        .arg("a")
        .env("RUST_LIB_BACKTRACE", "0")
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap();
    assert!(p.status.success());
    assert!(String::from_utf8_lossy(&p.stdout).contains("disabled backtrace\n"));

    let p = Command::new(me)
        .arg("force")
        .env("RUST_LIB_BACKTRACE", "0")
        .env("RUST_BACKTRACE", "0")
        .output()
        .unwrap();
    assert!(p.status.success());
    assert!(String::from_utf8_lossy(&p.stdout).contains("stack backtrace:\n"));

    let p = Command::new(me).arg("force").output().unwrap();
    assert!(p.status.success());
    assert!(String::from_utf8_lossy(&p.stdout).contains("stack backtrace:\n"));
}
