#![feature(associated_type_defaults)]

// Tests that a trait with one defaulted and one non-defaulted assoc. type behaves properly.

trait Trait {
    type Foo = u8;
    type Bar;
}

// `Bar` must be specified
impl Trait for () {}
//~^ error: not all trait items implemented, missing: `Bar`

impl Trait for bool {
//~^ error: not all trait items implemented, missing: `Bar`
    type Foo = ();
}

impl Trait for u8 {
    type Bar = ();
}

impl Trait for u16 {
    type Foo = String;
    type Bar = bool;
}

fn main() {
    let _: <u8 as Trait>::Foo = 0u8;
    let _: <u8 as Trait>::Bar = ();

    let _: <u16 as Trait>::Foo = String::new();
    let _: <u16 as Trait>::Bar = true;
}
