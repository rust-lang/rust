//@ check-pass
#![feature(const_block_items)]

macro_rules! foo {
    ($item:item) => {
        $item
    };
}

foo!(const {});

fn main() {}
