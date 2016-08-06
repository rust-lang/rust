// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test incremental compilation tracking where we change field names
// in between revisions (hashing should be stable).

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

#[cfg(rpass1)]
pub struct X {
    pub x: u32,
    pub x2: u32,
}

#[cfg(rpass2)]
pub struct X {
    pub x: u32,
}

pub struct EmbedX {
    x: X
}

pub struct Y {
    pub y: char
}

#[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
pub fn use_X(x: X) -> u32 {
    x.x as u32
}

#[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
pub fn use_EmbedX(embed: EmbedX) -> u32 {
    embed.x.x as u32
}

#[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
pub fn use_Y() {
    let x: Y = Y { y: 'c' };
}

pub fn main() { }
