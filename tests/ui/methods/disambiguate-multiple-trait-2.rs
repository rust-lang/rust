trait A {
    type Type; //~ NOTE ambiguous `Type` from `A`
    const CONST: usize = 1; //~ NOTE candidate #1
    fn foo(&self); //~ NOTE candidate #1
}

trait B {
    type Type; //~ NOTE ambiguous `Type` from `B`
    const CONST: usize; //~ NOTE candidate #2
    fn foo(&self); //~ NOTE candidate #2
}

trait C: A + B {}

fn a<T: C>(t: T) {
    t.foo(); //~ ERROR multiple applicable items in scope
    //~^ NOTE multiple `foo` found
    //~| HELP disambiguate the method
    //~| HELP disambiguate the method
    let _ = T::CONST; //~ ERROR multiple applicable items in scope
    //~^ NOTE multiple `CONST` found
    //~| HELP use fully-qualified syntax
    let _: T::Type; //~ ERROR ambiguous associated type
    //~^ NOTE ambiguous associated type `Type`
    //~| HELP use fully-qualified syntax
    //~| HELP use fully-qualified syntax
}

#[derive(Debug)]
struct S;

impl<T: std::fmt::Debug> A for T {
    type Type = ();
    const CONST: usize = 1; //~ NOTE candidate #1
    fn foo(&self) {} //~ NOTE candidate #1
}

impl<T: std::fmt::Debug> B for T {
    type Type = ();
    const CONST: usize = 1; //~ NOTE candidate #2
    fn foo(&self) {} //~ NOTE candidate #2
}

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
