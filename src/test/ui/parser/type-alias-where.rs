// check-fail

#![feature(generic_associated_types)]

// Fine, but lints as unused
type Foo where u32: Copy = ();
// Not fine.
type Bar = () where u32: Copy;
//~^ ERROR where clauses are not allowed
type Baz = () where;
//~^ ERROR where clauses are not allowed

trait Trait {
    // Fine.
    type Assoc where u32: Copy;
    // Fine.
    type Assoc2 where u32: Copy, i32: Copy;
}

impl Trait for u32 {
    // Not fine, suggests moving.
    type Assoc where u32: Copy = ();
    //~^ ERROR where clause not allowed here
    // Not fine, suggests moving `u32: Copy`
    type Assoc2 where u32: Copy = () where i32: Copy;
    //~^ ERROR where clause not allowed here
}

impl Trait for i32 {
    // Fine.
    type Assoc = () where u32: Copy;
    // Not fine, suggests moving both.
    type Assoc2 where u32: Copy, i32: Copy = ();
    //~^ ERROR where clause not allowed here
}

fn main() {}
