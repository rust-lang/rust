//! Test that print!/println! output to stdout and eprint!/eprintln!
//! output to stderr correctly.

//@ run-pass
//@ needs-subprocess

use std::{env, process};

fn child() {
    print!("[stdout 0]");
    print!("[stdout {}]", 1);
    println!("[stdout {}]", 2);
    println!();
    eprint!("[stderr 0]");
    eprint!("[stderr {}]", 1);
    eprintln!("[stderr {}]", 2);
    eprintln!();
}

fn parent() {
    let this = env::args().next().unwrap();
    let output = process::Command::new(this).arg("-").output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();

    assert_eq!(stdout, "[stdout 0][stdout 1][stdout 2]\n\n");
    assert_eq!(stderr, "[stderr 0][stderr 1][stderr 2]\n\n");
}

fn main() {
    if env::args().count() == 2 { child() } else { parent() }
}
