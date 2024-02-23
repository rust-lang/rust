#![crate_type = "lib"]
#![feature(unnamed_fields)]
#![allow(unused, incomplete_features)]

enum K {
    M {
        _ : struct { field: u8 },
        //~^ error: unnamed fields are not allowed outside of structs or unions
        //~| error: anonymous structs are not allowed outside of unnamed struct or union fields
    }
}
