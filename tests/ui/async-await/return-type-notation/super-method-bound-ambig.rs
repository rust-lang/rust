//@ edition:2021

#![feature(return_type_notation)]

trait Super1<'a> {
    async fn test();
}
impl Super1<'_> for () {
    async fn test() {}
}

trait Super2 {
    async fn test();
}
impl Super2 for () {
    async fn test() {}
}

trait Foo: for<'a> Super1<'a> + Super2 {}
impl Foo for () {}

fn test<T>()
where
    T: Foo<test(..): Send>,
    //~^ ERROR ambiguous associated function `test` in bounds of `Foo`
{
}

fn main() {
    test::<()>();
}
