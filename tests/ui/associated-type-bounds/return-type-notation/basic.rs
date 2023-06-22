// revisions: current_with current_without next_with next_without
// [next_with] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// [next_without] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// edition: 2021
// [current_with] check-pass
// [next_with] check-pass

#![feature(return_type_notation, async_fn_in_trait)]
//~^ WARN the feature `return_type_notation` is incomplete

trait Foo {
    async fn method() -> Result<(), ()>;
}

async fn foo<T: Foo>() -> Result<(), ()> {
    T::method().await?;
    Ok(())
}

fn is_send(_: impl Send) {}

fn test<
    #[cfg(any(current_with, next_with))] T: Foo<method(): Send>,
    #[cfg(any(current_without, next_without))] T: Foo,
>() {
    is_send(foo::<T>());
    //[current_without]~^ ERROR future cannot be sent between threads safely
    //[next_without]~^^ ERROR future cannot be sent between threads safely
}

fn main() {}
