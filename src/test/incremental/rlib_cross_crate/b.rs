// Same test as `type_alias_cross_crate`, but with
// `no-prefer-dynamic`, ensuring that we test what happens when we
// build rlibs (before we were only testing dylibs, which meant we
// didn't realize we had to preserve a `bc` file as well).

// aux-build:a.rs
// revisions:rpass1 rpass2 rpass3
// no-prefer-dynamic
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

extern crate a;

#[rustc_dirty(label="typeck_tables_of", cfg="rpass2")]
#[rustc_clean(label="typeck_tables_of", cfg="rpass3")]
pub fn use_X() -> u32 {
    let x: a::X = 22;
    x as u32
}

#[rustc_clean(label="typeck_tables_of", cfg="rpass2")]
#[rustc_clean(label="typeck_tables_of", cfg="rpass3")]
pub fn use_Y() {
    let x: a::Y = 'c';
}

pub fn main() { }
