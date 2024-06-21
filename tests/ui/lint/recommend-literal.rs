type Real = double;
//~^ ERROR cannot find type `double`
//~| HELP perhaps you intended to use this type

fn main() {
    let x: Real = 3.5;
    let y: long = 74802374902374923;
    //~^ ERROR cannot find type `long`
    //~| HELP perhaps you intended to use this type
    let v1: Boolean = true;
    //~^ ERROR: cannot find type `Boolean` [E0412]
    //~| HELP perhaps you intended to use this type
    let v2: Bool = true;
    //~^ ERROR: cannot find type `Bool` [E0412]
    //~| HELP a builtin type with a similar name exists
    //~| HELP perhaps you intended to use this type
}

fn z(a: boolean) {
    //~^ ERROR cannot find type `boolean`
    //~| HELP perhaps you intended to use this type
}

fn a() -> byte {
//~^ ERROR cannot find type `byte`
//~| HELP perhaps you intended to use this type
    3
}

struct Data { //~ HELP you might be missing a type parameter
    width: float,
    //~^ ERROR cannot find type `float`
    //~| HELP perhaps you intended to use this type
    depth: Option<int>,
    //~^ ERROR cannot find type `int`
    //~| HELP perhaps you intended to use this type
}

trait Stuff {}
impl Stuff for short {}
//~^ ERROR cannot find type `short`
//~| HELP perhaps you intended to use this type
