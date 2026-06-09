#![feature(associated_type_defaults)]

trait A {
    type Type = ();
    const CONST: usize = 1; //~ NOTE candidate #1
    fn foo(&self) {} //~ NOTE candidate #1
}

trait B {
    type Type = ();
    const CONST: usize = 2; //~ NOTE candidate #2
    fn foo(&self) {} //~ NOTE candidate #2
}

#[derive(Debug)]
struct S;

impl<T: std::fmt::Debug> A for T {}

impl<T: std::fmt::Debug> B for T {}

fn main() {
    let s = S;
    S::foo(&s); //~ ERROR multiple applicable items in scope
    //~^ NOTE multiple `foo` found
    //~| HELP use fully-qualified syntax
    let _ = S::CONST; //~ ERROR multiple applicable items in scope
    //~^ NOTE multiple `CONST` found
    //~| HELP use fully-qualified syntax
    let _: S::Type; //~ ERROR ambiguous associated type
    //~^ HELP use fully-qualified syntax
}
