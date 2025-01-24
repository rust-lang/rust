//@ run-rustfix
#![deny(unused_parens)]
#![allow(unreachable_code)]

fn foo() {
    loop {
        break (_ = 42);
        // lint unused_parens should not be triggered here.
    }

    let _ = loop {
        let a = 1;
        let b = 2;
        break (a + b); //~ERROR unnecessary parentheses
    };

    loop {
        if (break return ()) {
            //~^ ERROR unnecessary parentheses
        }
        if break (return ()) {
            //~^ ERROR unnecessary parentheses
        }
    }

    return (_ = 42);
    // lint unused_parens should not be triggered here.
}

fn main() {
    let _ = foo();
}
