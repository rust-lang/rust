type Real = double;
//~^ ERROR cannot find type `double` in this scope
//~| HELP `f64` primitive type

fn main() {
    let x: Real = 3.5;
    let y: long = 74802374902374923;
    //~^ ERROR cannot find type `long` in this scope
    //~| HELP `i64` primitive type
    let v1: Boolean = true;
    //~^ ERROR: cannot find type `Boolean` in this scope [E0425]
    //~| HELP `bool` primitive type
    let v2: Bool = true;
    //~^ ERROR: cannot find type `Bool` in this scope [E0425]
    //~| HELP a builtin type with a similar name exists
    //~| HELP `bool` primitive type
    //~| HELP: there is an enum variant `std::mem::type_info::TypeKind::Bool`; try using the variant's enum
}

fn z(a: boolean) {
    //~^ ERROR cannot find type `boolean` in this scope
    //~| HELP `bool` primitive type
}

fn a() -> byte {
//~^ ERROR cannot find type `byte` in this scope
//~| HELP `u8` primitive type
    3
}

struct Data { //~ HELP you might be missing a type parameter
    width: float,
    //~^ ERROR cannot find type `float` in this scope
    //~| HELP `f32` primitive type
    depth: Option<int>,
    //~^ ERROR cannot find type `int` in this scope
    //~| HELP `i32` primitive type
}

trait Stuff {}
impl Stuff for short {}
//~^ ERROR cannot find type `short` in this scope
//~| HELP `i16` primitive type
