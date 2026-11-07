//@ run-pass
trait Trait {
    type Type;
}

impl<'a> Trait for &'a () {
    type Type = u32;
}

fn foo<'a>(t: <&'a () as Trait>::Type) -> <&'a () as Trait>::Type {
    t
}

fn main() {
    assert_eq!(foo(4), 4);
}
