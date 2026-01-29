//@ check-pass
#![feature(const_items_unit_type_default)]

#[cfg(false)]
const _ = assert!(false);

#[expect(unused)]
const _ = {
    let a = 1;
    assert!(true);
};

fn main() {}
