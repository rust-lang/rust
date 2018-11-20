#![feature(never_type)]

#![deny(unreachable_code)]
#![allow(dead_code)]

fn never() -> ! {
    unimplemented!()
}

fn foo() -> bool {
    // Explicit matches on the never type are unwarned.
    match never() {}
    // But matches in unreachable code are warned.
    match never() {} //~ ERROR unreachable expression
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
