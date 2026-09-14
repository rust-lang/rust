//@ aux-build:a.rs
//@ revisions:rpass1 rpass2
//@ compile-flags: -Z query-dep-graph
//@ ignore-backends: gcc

#![feature(rustc_attrs)]

extern crate a;

use a::*;

#[rustc_clean(except="typeck_root", cfg="rpass2")]
pub fn use_x() -> u32 {
    let x: X = X { x: 22 };
    x.x as u32
}

#[rustc_clean(except="typeck_root", cfg="rpass2")]
pub fn use_embed_x(embed: EmbedX) -> u32 {
    embed.x.x as u32
}

#[rustc_clean(cfg="rpass2")]
pub fn use_y() {
    let x: Y = Y { y: 'c' };
}

pub fn main() { }
