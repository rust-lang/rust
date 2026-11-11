// Check that coroutines respect the muatability of their upvars.

#![feature(coroutines)]

fn mutate_upvar() {
    let x = 0;

    #[coroutine]
    move || {
        x = 1;
        //~^ ERROR
        yield;
    };
}

fn main() {}
