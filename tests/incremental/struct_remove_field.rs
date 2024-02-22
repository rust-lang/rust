// Test incremental compilation tracking where we change field names
// in between revisions (hashing should be stable).

//@ revisions:rpass1 rpass2
//@ compile-flags: -Z query-dep-graph

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

#[rustc_clean(except="typeck,fn_sig", cfg="rpass2")]
pub fn use_X(x: X) -> u32 {
    x.x as u32
}

#[rustc_clean(except="typeck", cfg="rpass2")]
pub fn use_EmbedX(embed: EmbedX) -> u32 {
    embed.x.x as u32
}

#[rustc_clean(cfg="rpass2")]
pub fn use_Y() {
    let x: Y = Y { y: 'c' };
}

pub fn main() { }
