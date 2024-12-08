//@ run-pass

// To avoid having to `or` gate `_` as an expr.
#![feature(generic_arg_infer)]

fn foo() -> [u8; 3] {
    let x: [u8; _] = [0; _];
    x
}

fn main() {
    assert_eq!([0; _], foo());
}
