//@ check-fail

#![feature(const_items_unit_type_default)]

const _ = {
    assert!(true);
    2 + 2 //~ ERROR: mismatched types [E0308]
};

const fn id<T>(t: T) -> T {
    t
}

const _ = id(2);
//~^ ERROR: mismatched types [E0308]
const _ = id(());


fn main() {}
