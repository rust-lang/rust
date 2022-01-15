// Regression test for issue #88393.

#![allow(deprecated, invalid_value, unused_must_use)]
#![forbid(unreachable_code)]

enum Void {}

fn foo() {
    if false {
        unsafe { std::mem::uninitialized::<Void>(); }
    }
    println!();
}

fn bar() {
    let b = false;
    match b {
        false => unsafe { std::mem::uninitialized::<Void>(); }
        _ => unreachable!(),
    }
    println!();
    //~^ ERROR: unreachable expression
}

fn main() {}
