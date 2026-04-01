//! Test that a diverging function as the final expression in a block does not
//! raise an 'unreachable code' lint.

//@ check-pass
#![deny(unreachable_code)]

enum Never {}

fn make_never() -> Never {
    loop {}
}

fn func() {
    make_never();
}

fn block() {
    {
        make_never();
    }
}

fn branchy() {
    if false {
        make_never();
    } else {
        make_never();
    }
}

fn main() {
    func();
    block();
}
