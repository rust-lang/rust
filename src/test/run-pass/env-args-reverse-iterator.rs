// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten

use std::env::args;
use std::process::Command;

fn assert_reverse_iterator_for_program_arguments(program_name: &str) {
    let args: Vec<_> = args().rev().collect();

    assert!(args.len() == 4);
    assert_eq!(args[0], "c");
    assert_eq!(args[1], "b");
    assert_eq!(args[2], "a");
    assert_eq!(args[3], program_name);

    println!("passed");
}

fn main() {
    let mut args = args();
    let me = args.next().unwrap();

    if let Some(_) = args.next() {
        assert_reverse_iterator_for_program_arguments(&me);
        return
    }

    let output = Command::new(&me)
        .arg("a")
        .arg("b")
        .arg("c")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"passed\n");
}
