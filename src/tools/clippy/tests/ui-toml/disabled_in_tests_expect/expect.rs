//@check-pass
//@compile-flags: --test

#![allow(clippy::unnecessary_literal_unwrap)]
#![warn(clippy::unwrap_used)]

fn main() {}

fn foo(opt: Option<i32>) {
    #[expect(clippy::unwrap_used)]
    opt.unwrap();
}

#[test]
fn unwrap_some() {
    Some(()).unwrap();
}
