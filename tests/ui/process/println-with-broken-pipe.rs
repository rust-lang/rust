//@ run-pass
//@ check-run-results
//@ needs-subprocess
//@ ignore-windows
//@ ignore-fuchsia
//@ ignore-horizon
//@ ignore-android
//@ ignore-ios no 'head'
//@ ignore-tvos no 'head'
//@ ignore-watchos no 'head'
//@ ignore-visionos no 'head'
//@ ignore-backends: gcc
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:CC"
//@ compile-flags: -Zon-broken-pipe=error

// Test what the error message looks like when `println!()` panics because of
// `std::io::ErrorKind::BrokenPipe`

use std::env;
use std::process::{Command, Stdio};

fn main() {
    let mut args = env::args();
    let me = args.next().unwrap();

    if let Some(arg) = args.next() {
        // More than enough iterations to fill any pipe buffer. Normally this
        // loop will end with a panic more or less immediately.
        for _ in 0..65536 * 64 {
            println!("{arg}");
        }
        unreachable!("should have panicked because of BrokenPipe");
    }

    // Set up a pipeline with a short-lived consumer and wait for it to finish.
    // This will produce the `println!()` panic message on stderr.
    let mut producer = Command::new(&me)
        .arg("this line shall appear exactly once on stdout")
        .env("RUST_BACKTRACE", "0")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut consumer =
        Command::new("head").arg("-n1").stdin(producer.stdout.take().unwrap()).spawn().unwrap();
    consumer.wait().unwrap();
    producer.wait().unwrap();
}
