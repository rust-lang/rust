#![allow(incomplete_features)]
#![feature(unnamed_fields)]

struct F {
    field: struct { field: u8 }, //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    _: struct { field: u8 },
}

struct G {
    _: (u8, u8), //~ ERROR unnamed fields can only have struct or union types
}

union H {
    field: struct { field: u8 }, //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    _: struct { field: u8 },
}

union I {
    _: (u8, u8), //~ ERROR unnamed fields can only have struct or union types
}

enum K {
    M {
        _ : struct { field: u8 }, //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
        //~^ ERROR unnamed fields are not allowed outside of structs or unions
    },
    N {
        _ : u8, //~ ERROR unnamed fields are not allowed outside of structs or unions
    }
}

fn main() {}
