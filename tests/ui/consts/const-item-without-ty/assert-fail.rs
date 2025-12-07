//@ check-fail

#![feature(const_items_unit_type_default)]

const _ = assert!(false);
//~^ ERROR: evaluation panicked: assertion failed: false [E0080]
const _ = assert!(2 + 2 == 5);
//~^ ERROR: evaluation panicked: assertion failed: 2 + 2 == 5 [E0080]

fn main() {}
