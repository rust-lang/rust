//@ run-rustfix

// check to make sure that we suggest adding `move` after `static`

#![feature(coroutines)]

fn check() -> impl Sized {
    let x = 0;
    #[coroutine]
    static || {
        //~^ ERROR E0373
        yield;
        x
    }
}

fn main() {
    check();
}
