// Regression test for <https://github.com/rust-lang/rust/issues/91514>.
//
// An associated const initialized with `panic!()`, referenced from a `match` arm
// with arms on both sides, used to ICE during codegen. It now fails const-eval
// cleanly instead. The failure only surfaces on a full build (not `check`), since
// the const is evaluated during codegen.

//@ build-fail

#![allow(path_statements)]

struct S;

impl S {
    const CONST: u8 = panic!(); //~ ERROR evaluation panicked: explicit panic
}

fn f(_: Option<()>, _: Option<u8>) {}

fn main() {
    match 0 {
        0 => {
            f(None, None);
        }
        1 => {
            S::CONST;
        }
        _ => {}
    };
}
