//@ edition:2021

#![feature(async_closure)]

fn main() {
    fn needs_async_fn(_: impl async Fn()) {}

    let mut x = 1;
    needs_async_fn(async || {
        //~^ ERROR expected a closure that implements the `async Fn` trait, but this closure only implements `async FnMut`
        x += 1;
    });

    let x = String::new();
    needs_async_fn(move || async move {
        //~^ ERROR expected a closure that implements the `async Fn` trait, but this closure only implements `async FnOnce`
        println!("{x}");
    });
}
