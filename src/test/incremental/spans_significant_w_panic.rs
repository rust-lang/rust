// This test makes sure that just changing a definition's location in the
// source file also changes its incr. comp. hash, if debuginfo is enabled.

// revisions:rpass1 rpass2

// compile-flags: -C overflow-checks=on

#![feature(rustc_attrs)]

#[cfg(rpass1)]
pub fn main() {
    let _ = 0u8 + 1;
}

#[cfg(rpass2)]
#[rustc_dirty(label="optimized_mir", cfg="rpass2")]
pub fn main() {
    let _ = 0u8 + 1;
}
