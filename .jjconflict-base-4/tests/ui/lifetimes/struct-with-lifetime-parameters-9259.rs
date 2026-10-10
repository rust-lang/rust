// https://github.com/rust-lang/rust/issues/9259
//@ run-pass
#![allow(dead_code)]

struct A<'a> {
    a: &'a [String],
    b: Option<&'a [String]>,
}

pub fn main() {
    let b: &[String] = &["foo".to_string()];
    let a = A {
        a: &["test".to_string()],
        b: Some(b),
    };
    assert_eq!(a.b.as_ref().unwrap()[0], "foo");
}
