type Real = double;
//~^ ERROR cannot find type `double` in this scope
//~| HELP perhaps you intended to use this type

fn main() {
    let x: Real = 3.5;
    let y: long = 74802374902374923;
    //~^ ERROR cannot find type `long` in this scope
    //~| HELP perhaps you intended to use this type
    let v1: Boolean = true;
    //~^ ERROR: cannot find type `Boolean` in this scope [E0412]
    //~| HELP perhaps you intended to use this type
    let v2: Bool = true;
    //~^ ERROR: cannot find type `Bool` in this scope [E0412]
    //~| HELP a builtin type with a similar name exists
    //~| HELP perhaps you intended to use this type
}

fn z(a: boolean) {
    //~^ ERROR cannot find type `boolean` in this scope
    //~| HELP perhaps you intended to use this type
}

fn a() -> byte {
//~^ ERROR cannot find type `byte` in this scope
//~| HELP perhaps you intended to use this type
    3
}

struct Data { //~ HELP you might be missing a type parameter
    width: float,
    //~^ ERROR cannot find type `float` in this scope
    //~| HELP perhaps you intended to use this type
    depth: Option<int>,
    //~^ ERROR cannot find type `int` in this scope
    //~| HELP perhaps you intended to use this type
}

trait Stuff {}
impl Stuff for short {}
//~^ ERROR cannot find type `short` in this scope
//~| HELP perhaps you intended to use this type
