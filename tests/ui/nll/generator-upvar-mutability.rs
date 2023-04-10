// Check that generators respect the mutability of their upvars.

#![feature(generators)]

fn mutate_upvar() {
    let x = 0;
    move || {
        x = 1;
        //~^ ERROR
        yield;
    };
}

fn main() {}
