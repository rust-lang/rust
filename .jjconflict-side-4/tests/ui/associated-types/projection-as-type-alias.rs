//! Regression test for <https://github.com/rust-lang/rust/issues/28828>.
//! This failed to compile as associated types aliases were not normalized.
//@ run-pass

pub trait Foo {
    type Out;
}

impl Foo for () {
    type Out = bool;
}

fn main() {
    type Bool = <() as Foo>::Out;

    let x: Bool = true;
    assert!(x);

    let y: Option<Bool> = None;
    assert_eq!(y, None);
}
