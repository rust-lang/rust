// aux-build:pub_hygiene.rs

extern crate pub_hygiene;

use pub_hygiene::*;

fn main() {
    let x = MyStruct {};
    //~^ ERROR cannot find struct, variant or union type `MyStruct` in this scope
}
