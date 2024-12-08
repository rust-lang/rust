trait A {
    type Type;
    const CONST: usize;
    fn foo(&self);
}

trait B {
    type Type;
    const CONST: usize;
    fn foo(&self);
}

struct S;

impl A for S {
    type Type = ();
    const CONST: usize = 1; //~ NOTE candidate #1
    fn foo(&self) {} //~ NOTE candidate #1
}

impl B for S {
    type Type = ();
    const CONST: usize = 2; //~ NOTE candidate #2
    fn foo(&self) {} //~ NOTE candidate #2
}

fn main() {
    let s = S;
    S::foo(&s); //~ ERROR multiple applicable items in scope
    //~^ NOTE multiple `foo` found
    //~| HELP use fully-qualified syntax
    let _: S::Type = (); //~ ERROR ambiguous associated type
    //~| HELP use fully-qualified syntax
    let _ = S::CONST; //~ ERROR multiple applicable items in scope
    //~^ NOTE multiple `CONST` found
    //~| HELP use fully-qualified syntax
}
