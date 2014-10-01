// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::{process, Command};
use std::os;

fn main() {
    let len = os::args().len();

    if len == 1 {
        test();
    } else {
        assert_eq!(len, 3);
    }
}

fn test() {
    let status = Command::new(os::self_exe_name().unwrap())
                         .arg("foo").arg("")
                         .stdout(process::InheritFd(1))
                         .stderr(process::InheritFd(2))
                         .status().unwrap();
    assert!(status.success());
}
