// https://github.com/rust-lang/rust/issues/7344
//@ run-pass
#![allow(unused_must_use)]

#![allow(unreachable_code)]

fn foo() -> bool { false }

fn bar() {
    return;
    !foo();
}

fn baz() {
    return;
    if "" == "" {}
}

pub fn main() {
    bar();
    baz();
}
