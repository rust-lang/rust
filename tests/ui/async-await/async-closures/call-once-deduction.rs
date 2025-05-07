//@ edition: 2021
//@ check-pass

#![feature(async_fn_traits, unboxed_closures)]

fn bar<F, O>(_: F)
where
    F: AsyncFnOnce<(), CallOnceFuture = O>,
{
}

fn main() {
    bar(async move || {});
}
