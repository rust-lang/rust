// issue: rust-lang/rust#126725

trait Foo {
    fn foo<'a>() -> <&'a impl Sized as Bar>::Output;
    //~^ ERROR `impl Trait` is not allowed in paths
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
