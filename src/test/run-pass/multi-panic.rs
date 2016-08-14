// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten

fn check_for_no_backtrace(test: std::process::Output) {
    assert!(!test.status.success());
    let err = String::from_utf8_lossy(&test.stderr);
    let mut it = err.lines();

    assert_eq!(it.next().map(|l| l.starts_with("thread '<unnamed>' panicked at")), Some(true));
    assert_eq!(it.next(), Some("note: Run with `RUST_BACKTRACE=1` for a backtrace."));
    assert_eq!(it.next().map(|l| l.starts_with("thread 'main' panicked at")), Some(true));
    assert_eq!(it.next(), None);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "run_test" {
        let _ = std::thread::spawn(|| {
            panic!();
        }).join();

        panic!();
    } else {
        let test = std::process::Command::new(&args[0]).arg("run_test")
                                                       .env_remove("RUST_BACKTRACE")
                                                       .output()
                                                       .unwrap();
        check_for_no_backtrace(test);
        let test = std::process::Command::new(&args[0]).arg("run_test")
                                                       .env("RUST_BACKTRACE","0")
                                                       .output()
                                                       .unwrap();
        check_for_no_backtrace(test);
    }
}
