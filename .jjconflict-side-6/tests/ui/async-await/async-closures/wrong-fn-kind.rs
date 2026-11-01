//@ edition:2021

fn needs_async_fn(_: impl AsyncFn()) {}

fn a() {
    let mut x = 1;
    needs_async_fn(async || {
        //~^ ERROR cannot borrow `x` as mutable, as it is a captured variable in a `Fn` closure
        x += 1;
    });
}

fn b() {
    let x = String::new();
    needs_async_fn(move || async move {
        //~^ ERROR expected a closure that implements the `AsyncFn` trait, but this closure only implements `AsyncFnOnce`
        println!("{x}");
    });
}

fn main() {}
