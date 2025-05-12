//@no-rustfix: no suggestions

#![warn(clippy::useless_vec)]

// Regression test for <https://github.com/rust-lang/rust-clippy/issues/13692>.
fn foo() {
    // There should be no suggestion in this case.
    let _some_variable = vec![
        //~^ useless_vec
        1, 2, // i'm here to stay
        3, 4, // but this one going away ;-;
    ]; // that is life anyways
}

fn main() {}
