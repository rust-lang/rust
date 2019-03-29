// Test incremental compilation tracking where we change field names
// in between revisions (hashing should be stable).

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

pub struct X {
    pub x: u32,

    #[cfg(rpass2)]
    pub x2: u32,
}

pub struct EmbedX {
    x: X
}

pub struct Y {
    pub y: char
}

#[rustc_dirty(label="typeck_tables_of", cfg="rpass2")]
pub fn use_X(x: X) -> u32 {
    x.x as u32
}

#[rustc_dirty(label="typeck_tables_of", cfg="rpass2")]
pub fn use_EmbedX(embed: EmbedX) -> u32 {
    embed.x.x as u32
}

#[rustc_clean(label="typeck_tables_of", cfg="rpass2")]
pub fn use_Y() {
    let x: Y = Y { y: 'c' };
}

pub fn main() { }
