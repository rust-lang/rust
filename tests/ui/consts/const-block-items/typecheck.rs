//@ check-fail

#![feature(const_block_items)]

const {
    assert!(true);
    2 + 2 //~ ERROR: mismatched types [E0308]
}


const fn id<T>(t: T) -> T {
    t
}

const { id(2) }
//~^ ERROR: mismatched types [E0308]
const { id(()) }


fn main() {}
