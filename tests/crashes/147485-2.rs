//@ known-bug: #147485
//@ compile-flags: -g -O

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
