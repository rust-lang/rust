//@ run-pass

/*
#7519 ICE pattern matching unit in function argument
https://github.com/rust-lang/rust/issues/7519
*/

fn foo(():()) { }

pub fn main() {
    foo(());
}
