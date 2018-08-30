// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This used to segfault #30081

pub enum Instruction {
    Increment(i8),
    Loop(Box<Box<()>>),
}

fn main() {
    let instrs: Option<(u8, Box<Instruction>)> = None;
    instrs.into_iter()
        .map(|(_, instr)| instr)
        .map(|instr| match *instr { _other => {} })
        .last();
}
