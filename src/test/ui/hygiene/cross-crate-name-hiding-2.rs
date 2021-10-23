// aux-build:use_by_macro.rs

extern crate use_by_macro;

use use_by_macro::*;

my_struct!(define);

fn main() {
    let x = MyStruct {};
    //~^ ERROR cannot find struct, variant or union type `MyStruct` in this scope
}
