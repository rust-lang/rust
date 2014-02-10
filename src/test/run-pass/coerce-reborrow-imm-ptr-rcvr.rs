// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct SpeechMaker {
    speeches: uint
}

impl SpeechMaker {
    pub fn how_many(&self) -> uint { self.speeches }
}

fn foo(speaker: &SpeechMaker) -> uint {
    speaker.how_many() + 33
}

pub fn main() {
    let lincoln = SpeechMaker {speeches: 22};
    assert_eq!(foo(&lincoln), 55);
}
