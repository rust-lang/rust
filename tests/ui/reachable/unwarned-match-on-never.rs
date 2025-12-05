#![deny(unreachable_code)]
#![allow(dead_code)]

#![feature(never_type)]

fn foo(x: !) -> bool {
    // Explicit matches on the never type are unwarned.
    match x {}
    // But matches in unreachable code are warned.
    match x {} //~ ERROR unreachable expression
}

fn bar() {
    match (return) {
        () => () //~ ERROR unreachable arm
    }
}

fn main() {
    return;
    match () { //~ ERROR unreachable expression
        () => (),
    }
}
