#![feature(const_block_items)]

//@ check-pass

// FIXME(const_block_items): `pub`` is useless here
pub const {
    assert!(true);
}

fn main() { }
