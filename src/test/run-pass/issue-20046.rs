// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(PartialEq)]
enum IoErrorKind { BrokenPipe, XXX }
struct IoError {
    pub kind: IoErrorKind,
    pub detail: Option<String>
}
fn main() {
    let e: Result<u8, _> = Err(IoError{ kind: IoErrorKind::XXX, detail: None });
    match e {
        Ok(_) => true,
        Err(ref e) if e.kind == IoErrorKind::BrokenPipe => return,
        Err(IoError { kind: IoErrorKind::BrokenPipe, ..}) => return,
        Err(err) => panic!(err)
    };
}
