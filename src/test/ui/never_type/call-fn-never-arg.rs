// Test that we can use a ! for an argument of type !

// check-pass

#![feature(never_type)]
#![allow(unreachable_code)]

fn foo(x: !) -> ! {
    x
}

fn bar(x: never) -> never {
    x
}

fn main() {
    foo(panic!("wowzers!"));
    bar(panic!("wowzers!"))
}
