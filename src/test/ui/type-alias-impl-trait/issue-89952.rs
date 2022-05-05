// check-pass

#![feature(type_alias_impl_trait)]

trait SomeTrait {}
impl SomeTrait for () {}

trait MyFuture {
    type Output;
}
impl<T> MyFuture for T {
    type Output = T;
}

trait ReturnsFuture {
    type Output: SomeTrait;
    type Future: MyFuture<Output = Result<Self::Output, ()>>;
    fn func() -> Self::Future;
}

struct Foo;

impl ReturnsFuture for Foo {
    type Output = impl SomeTrait;
    type Future = impl MyFuture<Output = Result<Self::Output, ()>>;
    fn func() -> Self::Future {
        Result::<(), ()>::Err(())
    }
}

fn main() {}
