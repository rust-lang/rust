// check-pass
// run-rustfix

#![feature(associated_type_defaults)]

trait Trait {
    // Not fine, suggests moving.
    type Assoc where u32: Copy = ();
    //~^ WARNING where clause not allowed here
    // Not fine, suggests moving `u32: Copy`
    type Assoc2 where u32: Copy = () where i32: Copy;
    //~^ WARNING where clause not allowed here
}

fn main() {}
