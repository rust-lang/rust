// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Debug)]
pub enum Event {
    Key(u8),
    Resize,
    Unknown(u16),
}

static XTERM_SINGLE_BYTES : [(u8, Event); 1] = [(1,  Event::Resize)];

fn main() {
    match XTERM_SINGLE_BYTES[0] {
        (1, Event::Resize) => {},
        ref bad => panic!("unexpected {:?}", bad)
    }
}
