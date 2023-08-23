#![allow(incomplete_features)]
#![feature(unnamed_fields)]

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

enum K {
    M {
        _ : union { field: u8 }, //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
        //~^ ERROR unnamed fields are not allowed outside of structs or unions
        //~| ERROR anonymous unions are unimplemented
    },
    N {
        _ : u8, //~ ERROR unnamed fields are not allowed outside of structs or unions
    }
}

fn main() {}
