//@ edition:2021
//@ check-pass

#![feature(return_type_notation)]

trait Super<'a> {
    async fn test();
}
impl Super<'_> for () {
    async fn test() {}
}

trait Foo: for<'a> Super<'a> {}
impl Foo for () {}

fn test<T>()
where
    T: Foo<test(..): Send>,
{
}

fn main() {
    test::<()>();
}
