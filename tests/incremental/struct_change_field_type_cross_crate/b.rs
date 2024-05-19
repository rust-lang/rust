//@ aux-build:a.rs
//@ revisions:rpass1 rpass2
//@ compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

extern crate a;

use a::*;

#[rustc_clean(except="typeck", cfg="rpass2")]
pub fn use_X() -> u32 {
    let x: X = X { x: 22 };
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
