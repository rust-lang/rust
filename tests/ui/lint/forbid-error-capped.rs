//@ check-pass
// compile-args: --cap-lints=warn -Fwarnings

// This checks that the forbid attribute checking is ignored when the forbidden
// lint is capped.

#![forbid(warnings)]
#![allow(unused)]

#[allow(unused)]
mod bar {
    fn bar() {}
}

fn main() {}
