//@ check-pass
// Test for issue #101532 - dead code warnings should work inside const _

#![warn(dead_code)]

const _: () = {
    let a: ();
    struct B {} //~ WARN struct `B` is never constructed
    enum C {}   //~ WARN enum `C` is never used
    fn d() {}   //~ WARN function `d` is never used
    const E: () = {}; //~ WARN constant `E` is never used
};

fn main() {}
