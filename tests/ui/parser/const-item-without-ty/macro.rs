//@ check-pass
#![feature(const_items_unit_type_default)]

macro_rules! foo {
    ($item:item) => {
        $item
    };
}

foo!(const _ = {};);

fn main() {
    foo!(const _ = {};);
}
