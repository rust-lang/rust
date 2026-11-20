// Test that we can use a ! for an argument of type !
//
//@ check-pass

#![expect(unreachable_code)]

fn foo(x: !) -> ! {
    x
}

fn main() {
    foo(panic!("wowzers!"))
}
