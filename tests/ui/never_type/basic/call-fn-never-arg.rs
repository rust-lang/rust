// Test that we can use a ! for an argument of type !
//
//@ check-pass

#![feature(never_type)]
#![expect(unreachable_code)]

fn foo(x: !) -> ! {
    x
}

fn main() {
    foo(panic!("wowzers!"))
}
