// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::{ChanReader, ChanWriter};
use std::task::TaskBuilder;

fn main() {
    let (tx, rx) = channel();
    let mut reader = ChanReader::new(rx);
    let stderr = ChanWriter::new(tx);

    let res = TaskBuilder::new().stderr(box stderr as Box<Writer + Send>).try(proc() -> () {
        fail!("Hello, world!")
    });
    assert!(res.is_err());

    let output = reader.read_to_string().unwrap();
    assert!(output.as_slice().contains("Hello, world!"));
}
