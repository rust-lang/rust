// Regression test for issue: #47648
//@ edition:2024

struct Foo {}

fn call<F: FnOnce() -> Foo>(f: F) {
    call(&f);
    //~^ ERROR expected `FnOnce()` signature, found `&F`
}

fn async_call<F: AsyncFnOnce() -> Foo>(f: F) {
    async_call(&f);
    //~^ ERROR the trait bound `&F: AsyncFnOnce()` is not satisfied
}

fn main() {
    let foo = Foo {};
    let once = move || foo;
    //~^ ERROR expected a closure that implements the `Fn` trait, but this closure only implements `FnOnce` [E0525]
    //~| ERROR expected a closure that implements the `FnMut` trait, but this closure only implements `FnOnce` [E0525]

    call(&once);
    call(&mut once);

    let foo = Foo {};
    let once = &move || foo;
    //~^ ERROR expected a closure that implements the `Fn` trait, but this closure only implements `FnOnce` [E0525]

    call(once);

    let foo = Foo {};
    let once = &mut move || foo;
    //~^ ERROR expected a closure that implements the `FnMut` trait, but this closure only implements `FnOnce` [E0525]

    call(once);

    let foo = Foo {};
    let async_once = async move || foo;
    //~^ ERROR expected a closure that implements the `AsyncFn` trait, but this closure only implements `AsyncFnOnce` [E0525]
    //~| ERROR expected a closure that implements the `AsyncFnMut` trait, but this closure only implements `AsyncFnOnce` [E0525]

    async_call(&async_once);
    async_call(&mut async_once);

    let foo = Foo {};
    let async_once = &async move || foo;
    //~^ ERROR expected a closure that implements the `AsyncFn` trait, but this closure only implements `AsyncFnOnce` [E0525]

    async_call(async_once);

    let foo = Foo {};
    let async_once = &mut async move || foo;
    //~^ ERROR expected a closure that implements the `AsyncFnMut` trait, but this closure only implements `AsyncFnOnce` [E0525]

    async_call(async_once);
}
