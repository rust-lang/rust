//@ known-bug: rust-lang/rust#126725
trait Foo {
    fn foo<'a>(&'a self) -> <&'a impl Sized as Bar>::Output;
}

trait Bar {
    type Output;
}

struct X(i32);

impl<'a> Bar for &'a X {
    type Output = &'a i32;
}

impl Foo for X {
    fn foo<'a>(&'a self) -> <&'a Self as Bar>::Output {
        &self.0
    }
}
