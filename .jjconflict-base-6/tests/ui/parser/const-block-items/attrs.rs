//@ check-pass

#![feature(const_block_items)]

#[cfg(false)]
const { assert!(false) }

#[expect(unused)]
const {
    let a = 1;
    assert!(true);
}

fn main() {}
