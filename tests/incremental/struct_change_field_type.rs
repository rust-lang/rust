// Test incremental compilation tracking where we change nothing
// in between revisions (hashing should be stable).

//@ revisions:rpass1 rpass2
//@ compile-flags: -Z query-dep-graph
//@ ignore-backends: gcc

#![feature(rustc_attrs)]

#[cfg(rpass1)]
pub struct X {
    pub x: u32
}

#[cfg(rpass2)]
pub struct X {
    pub x: i32
}

pub struct EmbedX {
    x: X
}

pub struct Y {
    pub y: char
}

#[rustc_clean(except="typeck_root", cfg="rpass2")]
pub fn use_x() -> u32 {
    let x: X = X { x: 22 };
    x.x as u32
}

#[rustc_clean(except="typeck_root", cfg="rpass2")]
pub fn use_embed_x(x: EmbedX) -> u32 {
    let x: X = X { x: 22 };
    x.x as u32
}

#[rustc_clean(cfg="rpass2")]
pub fn use_y() {
    let x: Y = Y { y: 'c' };
}

pub fn main() { }
