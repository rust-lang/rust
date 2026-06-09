//@ check-pass

#![allow(clippy::unit_arg, clippy::no_effect)]

const fn v(_: ()) {}

fn main() {
    if true {
        v({
            [0; 1 + 1];
        });
        Some(())
    } else {
        None
    };
}
