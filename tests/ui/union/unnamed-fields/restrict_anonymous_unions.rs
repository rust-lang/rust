#![allow(incomplete_features)]
#![feature(unnamed_fields)]

fn f() -> union { field: u8 } {} //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous unions are unimplemented

fn f2(a: union { field: u8 } ) {} //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous unions are unimplemented

struct F {
    field: union { field: u8 }, //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous unions are unimplemented
    _: union { field: u8 },
    //~^ ERROR anonymous unions are unimplemented
}

struct G {
    _: (u8, u8), //~ ERROR unnamed fields can only have struct or union types
}

union H {
    field: union { field: u8 }, //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous unions are unimplemented
    _: union { field: u8 },
    //~^ ERROR anonymous unions are unimplemented
}

union I {
    _: (u8, u8), //~ ERROR unnamed fields can only have struct or union types
}

struct J(union { field: u8 }, u8); //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous unions are unimplemented

enum K {
    L(union { field: u8 }), //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous unions are unimplemented
    M {
        _ : union { field: u8 }, //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
        //~^ ERROR unnamed fields are not allowed outside of structs or unions
        //~| ERROR anonymous unions are unimplemented
    },
    N {
        _ : u8, //~ ERROR unnamed fields are not allowed outside of structs or unions
    }
}

const L: union { field: u8 } = 0; //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous unions are unimplemented

static M: union { field: u8 } = 0; //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous unions are unimplemented

type N = union { field: u8 }; //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous unions are unimplemented

impl union { field: u8 } {} //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
// //~^ ERROR anonymous unions are unimplemented

trait Foo {}

impl Foo for union { field: u8 } {} //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous unions are unimplemented

fn main() {
    let p: [union { field: u8 }; 1]; //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous unions are unimplemented

    let q: (union { field: u8 }, u8); //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous unions are unimplemented

    let c = || -> union { field: u8 } {}; //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous unions are unimplemented
}
