//@ run-pass
#![allow(unreachable_code)]

// Regression test for #39808. The type parameter of `Owned` was
// considered to be "unconstrained" because the type resulting from
// `format!` (`String`) was not being propagated upward, owing to the
// fact that the expression diverges.

use std::borrow::Cow;

fn main() {
    let _ = if false {
        Cow::Owned(format!("{:?}", panic!()))
    } else {
        Cow::Borrowed("")
    };
}
