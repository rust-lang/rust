// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Be sure that when a SIGPIPE would have been received that the entire process
// doesn't die in a ball of fire, but rather it's gracefully handled.

use std::os;
use std::io::{PipeStream, Process};

fn test() {
    let os::Pipe { input, out } = os::pipe();
    let input = PipeStream::open(input);
    let mut out = PipeStream::open(out);
    drop(input);

    let _ = out.write([1]);
}

fn main() {
    let args = os::args();
    let args = args.as_slice();
    if args.len() > 1 && args[1].as_slice() == "test" {
        return test();
    }

    let mut p = Process::new(args[0], ["test".to_owned()]).unwrap();
    assert!(p.wait().success());
}
