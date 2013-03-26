// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use super::Stream;

pub struct FileStream;

pub impl FileStream {
    fn new(_path: Path) -> FileStream {
        fail!()
    }
}

impl Stream for FileStream {
    fn read(&mut self, _buf: &mut [u8]) -> uint {
        fail!()
    }

    fn eof(&mut self) -> bool {
        fail!()
    }

    fn write(&mut self, _v: &const [u8]) {
        fail!()
    }
}

#[test]
#[ignore]
fn super_simple_smoke_test_lets_go_read_some_files_and_have_a_good_time() {
    let message = "it's alright. have a good time";
    let filename = Path("test.txt");
    let mut outstream = FileStream::new(filename);
    outstream.write(message.to_bytes());
}
