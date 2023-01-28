#![allow(incomplete_features)]
#![feature(unnamed_fields)]

fn f() -> struct { field: u8 } {} //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented

fn f2(a: struct { field: u8 } ) {} //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented


struct F {
    field: struct { field: u8 } //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented
}

union G {
    field1: struct { field: u8 } //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented
}

struct I(struct { field: u8 }, u8); //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented

enum J {
    K(struct { field: u8 }), //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented
    L {
        _ : struct { field: u8 } //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
        //~^ ERROR anonymous fields are not allowed outside of structs or unions
        //~| ERROR anonymous structs are unimplemented
    },
    M {
        _ : u8 //~ ERROR anonymous fields are not allowed outside of structs or unions
    }
}

const L: struct { field: u8 } = 0; //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented

static M: struct { field: u8 } = 0; //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented

type N = struct { field: u8 }; //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented

impl struct { field: u8 } {} //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented

trait Foo {}

impl Foo for struct { field: u8 } {} //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented

fn main() {
    let p: [struct { field: u8 }; 1]; //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented

    let q: (struct { field: u8 }, u8); //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented

    let c = || -> struct { field: u8 } {}; //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented
}
