// edition:2021

#![feature(async_fn_in_trait, return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

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
    T: Foo<test(): Send>,
    //~^ ERROR ambiguous associated function `test` for `Foo`
{
}

fn main() {
    test::<()>();
}
