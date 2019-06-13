// compile-pass

#![feature(associated_type_defaults)]

trait Tr {
    type Assoc = ();
}

impl Tr for () {}

fn f(thing: <() as Tr>::Assoc) {
    let c: () = thing;
}

fn main() {}
