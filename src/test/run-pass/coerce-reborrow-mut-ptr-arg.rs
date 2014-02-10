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

fn talk(x: &mut SpeechMaker) {
    x.speeches += 1;
}

fn give_a_few_speeches(speaker: &mut SpeechMaker) {

    // Here speaker is reborrowed for each call, so we don't get errors
    // about speaker being moved.

    talk(speaker);
    talk(speaker);
    talk(speaker);
}

pub fn main() {
    let mut lincoln = SpeechMaker {speeches: 22};
    give_a_few_speeches(&mut lincoln);
}
