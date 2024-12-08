trait Foo {
    type Type;

    fn foo();
    fn bar();
    fn qux();
}

struct A;

impl Foo for A {
//~^ ERROR not all trait items implemented
    type Typ = ();
    //~^ ERROR type `Typ` is not a member of trait
    //~| HELP there is an associated type with a similar name

    fn fooo() {}
    //~^ ERROR method `fooo` is not a member of trait
    //~| HELP there is an associated function with a similar name

    fn barr() {}
    //~^ ERROR method `barr` is not a member of trait
    //~| HELP there is an associated function with a similar name

    fn quux() {}
    //~^ ERROR method `quux` is not a member of trait
    //~| HELP there is an associated function with a similar name
}
//~^ HELP implement the missing item
//~| HELP implement the missing item
//~| HELP implement the missing item
//~| HELP implement the missing item

trait Bar {
    const Const: i32;
}

struct B;

impl Bar for B {
//~^ ERROR not all trait items implemented
    const Cnst: i32 = 0;
    //~^ ERROR const `Cnst` is not a member of trait
    //~| HELP there is an associated constant with a similar name
}
//~^ HELP implement the missing item

fn main() {}
