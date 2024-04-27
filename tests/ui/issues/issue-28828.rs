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
