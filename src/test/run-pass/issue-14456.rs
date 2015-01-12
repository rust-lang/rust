// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::old_io::process;
use std::old_io::Command;
use std::old_io;
use std::os;

fn main() {
    let args = os::args();
    if args.len() > 1 && args[1] == "child" {
        return child()
    }

    test();

}

fn child() {
    old_io::stdout().write_line("foo").unwrap();
    old_io::stderr().write_line("bar").unwrap();
    let mut stdin = old_io::stdin();
    assert_eq!(stdin.lock().read_line().err().unwrap().kind, old_io::EndOfFile);
}

fn test() {
    let args = os::args();
    let mut p = Command::new(&args[0]).arg("child")
                                     .stdin(process::Ignored)
                                     .stdout(process::Ignored)
                                     .stderr(process::Ignored)
                                     .spawn().unwrap();
    assert!(p.wait().unwrap().success());
}
