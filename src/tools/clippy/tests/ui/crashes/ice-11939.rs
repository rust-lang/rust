//@ check-pass

#![expect(clippy::no_effect, clippy::unit_arg)]

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
