//@ check-pass
//@ edition: 2018

#![allow(unused)]
#![warn(unreachable_code)]

macro_rules! later {
    () => { todo!() };
}

fn foo() {
    todo!();
    let this_is_unreachable = 1;
}

fn bar() {
    panic!("This is really unreachable");
    let really_unreachable = true;
    //~^ WARNING: unreachable
}

fn baz() -> bool {
    if true {
        todo!();
        false
    } else if todo!() {
        true
    } else {
        later!();
        false
    }
}

fn main() {
    foo();
    bar();
    if baz() {
        todo!();
    }
    let this_is_reachable = 1;
}
