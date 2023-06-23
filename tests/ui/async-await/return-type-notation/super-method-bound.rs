// edition:2021
// check-pass
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait, return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

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
    T: Foo<test(): Send>,
{
}

fn main() {
    test::<()>();
}
