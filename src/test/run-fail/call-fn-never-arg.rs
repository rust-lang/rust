// Test that we can use a ! for an argument of type !

// error-pattern:wowzers!

#![feature(never_type)]
#![allow(unreachable_code)]

fn foo(x: !) -> ! {
    x
}

fn main() {
    foo(panic!("wowzers!"))
}
