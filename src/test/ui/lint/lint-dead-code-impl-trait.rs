#![deny(dead_code)]

trait Trait {
    type Type;
}

impl Trait for () {
    type Type = ();
}

type Used = ();
type Unused = (); //~ ERROR type alias is never used

fn foo() -> impl Trait<Type = Used> {}

fn main() {
    foo();
}
