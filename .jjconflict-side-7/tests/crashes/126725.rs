//@ known-bug: #126725
type Indirect<T> = <T as Bar>::Output;

trait Foo {
    fn foo<'a>() -> Indirect<&'a impl Sized>;
}

trait Bar {
    type Output;
}

impl<'a> Bar for &'a () {
    type Output = &'a i32;
}

impl Foo for () {
    fn foo<'a>() -> <&'a Self as Bar>::Output {
        &0
    }
}

fn main() {}
