#![allow(incomplete_features)]
#![feature(unnamed_fields)]

fn f() -> struct { field: u8 } {} //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented

fn f2(a: struct { field: u8 } ) {} //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous structs are unimplemented

union G {
    field: struct { field: u8 } //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented
}
//~| ERROR unions may not contain fields that need dropping [E0740]

struct H { _: u8 } // Should error after hir checks

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

static M: union { field: u8 } = 0; //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous unions are unimplemented

type N = union { field: u8 }; //~ ERROR anonymous unions are not allowed outside of unnamed struct or union fields
//~^ ERROR anonymous unions are unimplemented

fn main() {
    const O: struct { field: u8 } = 0; //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented

    let p: [struct { field: u8 }; 1]; //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented

    let q: (struct { field: u8 }, u8); //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented

    let cl = || -> struct { field: u8 } {}; //~ ERROR anonymous structs are not allowed outside of unnamed struct or union fields
    //~^ ERROR anonymous structs are unimplemented
}
