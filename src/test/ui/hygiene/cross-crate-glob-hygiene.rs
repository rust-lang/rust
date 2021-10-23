// aux-build:use_by_macro.rs

extern crate use_by_macro;

use use_by_macro::*;

mod m {
    use use_by_macro::*;

    my_struct!(define);
}

use m::*;

fn main() {
    let x = my_struct!(create);
    //~^ ERROR cannot find struct, variant or union type `MyStruct` in this scope
}
