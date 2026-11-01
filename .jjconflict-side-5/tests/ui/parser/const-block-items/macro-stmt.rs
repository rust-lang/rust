//@ check-fail

#![feature(const_block_items)]

macro_rules! foo {
    ($item:item) => {
        $item
        //~^ ERROR: expected expression, found ``
    };
}

fn main() {
    foo!(const {});
}
