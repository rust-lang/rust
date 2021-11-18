// Test incremental compilation tracking where we change field names
// in between revisions (hashing should be stable).

// revisions:rpass1 cfail2
// compile-flags: -Z query-dep-graph
// [cfail2] compile-flags: -Z query-dep-graph -Z assert-incr-state=loaded

#![feature(rustc_attrs)]

#[cfg(rpass1)]
pub struct X {
    pub x: u32
}

#[cfg(cfail2)]
pub struct X {
    pub y: u32
}

pub struct EmbedX {
    x: X
}

pub struct Y {
    pub y: char
}

#[rustc_clean(except="typeck", cfg="cfail2")]
pub fn use_X() -> u32 {
    let x: X = X { x: 22 };
    //[cfail2]~^ ERROR struct `X` has no field named `x`
    x.x as u32
    //[cfail2]~^ ERROR no field `x` on type `X`
}

#[rustc_clean(except="typeck", cfg="cfail2")]
pub fn use_EmbedX(embed: EmbedX) -> u32 {
    embed.x.x as u32
    //[cfail2]~^ ERROR no field `x` on type `X`
}

#[rustc_clean(cfg="cfail2")]
pub fn use_Y() {
    let x: Y = Y { y: 'c' };
}

pub fn main() { }
