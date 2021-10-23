// check-pass
// aux-build:use_by_macro.rs

#![feature(type_name_of_val)]
extern crate use_by_macro;

use use_by_macro::*;

enum MyStruct {}
my_struct!(define);

fn main() {
    let x = my_struct!(create);
}
