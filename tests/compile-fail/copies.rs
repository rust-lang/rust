#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![deny(clippy)]

fn foo() -> bool { unimplemented!() }

fn if_same_then_else() {
    if true { //~ERROR this if has the same then and else blocks
        foo();
    }
    else {
        foo();
    }

    if true {
        foo();
        foo();
    }
    else {
        foo();
    }

    let _ = if true { //~ERROR this if has the same then and else blocks
        foo();
        42
    }
    else {
        foo();
        42
    };

    if true {
        foo();
    }

    let _ = if true { //~ERROR this if has the same then and else blocks
        42
    }
    else {
        42
    };
}

fn ifs_same_cond() {
    let a = 0;

    if a == 1 {
    }
    else if a == 1 { //~ERROR this if has the same condition as a previous if
    }

    if 2*a == 1 {
    }
    else if 2*a == 2 {
    }
    else if 2*a == 1 { //~ERROR this if has the same condition as a previous if
    }
    else if a == 1 {
    }

    // Ok, maybe `foo` isn’t pure and this actually makes sense. But you should probably refactor
    // this to make the intention clearer anyway.
    if foo() {
    }
    else if foo() { //~ERROR this if has the same condition as a previous if
    }
}

fn main() {}
