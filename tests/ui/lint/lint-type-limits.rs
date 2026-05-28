#![allow(dead_code)]

//@ compile-flags: -D unused-comparisons
fn main() { }

fn foo() {
    let mut i = 100_usize;
    while i >= 0 { //~ ERROR comparison is useless due to type limits
        i -= 1;
    }
}

fn bar() -> i8 {
    return 123;
}

fn bleh() {
    let u = 42u8;
    let _ = u > 255; //~ ERROR comparison is useless due to type limits
    let _ = 255 < u; //~ ERROR comparison is useless due to type limits
    let _ = u < 0; //~ ERROR comparison is useless due to type limits
    let _ = 0 > u; //~ ERROR comparison is useless due to type limits
    let _ = u <= 255; //~ ERROR comparison is useless due to type limits
    let _ = 255 >= u; //~ ERROR comparison is useless due to type limits
    let _ = u >= 0; //~ ERROR comparison is useless due to type limits
    let _ = 0 <= u; //~ ERROR comparison is useless due to type limits
}
