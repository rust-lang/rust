//@ build-pass
//@ compile-flags: -g -O

// Regression test for #147485.

#![crate_type = "lib"]

pub fn foo(a: bool, b: bool) -> bool {
    let mut c = &a;
    if false {
        return *c;
    }
    let d = b && a;
    if d {
        c = &b;
    }
    b
}
