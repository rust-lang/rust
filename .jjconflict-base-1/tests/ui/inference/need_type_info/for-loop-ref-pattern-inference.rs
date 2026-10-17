//! Regression test for <https://github.com/rust-lang/rust/issues/20261>.

fn main() {
    // N.B., this (almost) typechecks when default binding modes are enabled.
    for (ref i,) in [].iter() {
        i.clone();
        //~^ ERROR type annotations needed
    }
}
