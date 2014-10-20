// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::io::process;
use std::io::Command;
use std::io;
use std::os;

fn main() {
    let args = os::args();
    if args.len() > 1 && args[1].as_slice() == "child" {
        return child()
    }

    test();

}

fn child() {
    io::stdout().write_line("foo").unwrap();
    io::stderr().write_line("bar").unwrap();
    assert_eq!(io::stdin().read_line().err().unwrap().kind, io::EndOfFile);
}

fn test() {
    let args = os::args();
    let mut p = Command::new(args[0].as_slice()).arg("child")
                                     .stdin(process::Ignored)
                                     .stdout(process::Ignored)
                                     .stderr(process::Ignored)
                                     .spawn().unwrap();
    assert!(p.wait().unwrap().success());
}
