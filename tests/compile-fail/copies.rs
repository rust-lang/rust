#![feature(plugin)]
#![plugin(clippy)]

#![deny(clippy)]

fn foo() -> bool { unimplemented!() }

fn main() {
    let a = 0;

    if a == 1 {
    }
    else if a == 1 { //~ERROR this if as the same condition as a previous if
    }

    if 2*a == 1 {
    }
    else if 2*a == 2 {
    }
    else if 2*a == 1 { //~ERROR this if as the same condition as a previous if
    }
    else if a == 1 {
    }

    // Ok, maybe `foo` isn’t pure and this actually makes sense. But you should probably refactor
    // this to make the intention clearer anyway.
    if foo() {
    }
    else if foo() { //~ERROR this if as the same condition as a previous if
    }
}
