// Provide clear diagnostics when `&FnOnce` was provided
// when `FnOnce` was expected in function arguments.
// Related issue: #47648
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

    call(&once);
    //~^ ERROR expected `FnOnce`, found reference [E0525]
    call(&mut once);
    //~^ ERROR expected `FnOnce`, found reference [E0525]

    let foo = Foo {};
    let once = &move || foo;

    call(once);
    //~^ ERROR expected `FnOnce`, found reference [E0525]

    let foo = Foo {};
    let once = &mut move || foo;

    call(once);
    //~^ ERROR expected `FnOnce`, found reference [E0525]

    let foo = Foo {};
    let async_once = async move || foo;

    async_call(&async_once);
    //~^ ERROR expected `AsyncFnOnce`, found reference [E0525]
    async_call(&mut async_once);
    //~^ ERROR expected `AsyncFnOnce`, found reference [E0525]

    let foo = Foo {};
    let async_once = &async move || foo;

    async_call(async_once);
    //~^ ERROR expected `AsyncFnOnce`, found reference [E0525]

    let foo = Foo {};
    let async_once = &mut async move || foo;

    async_call(async_once);
    //~^ ERROR expected `AsyncFnOnce`, found reference [E0525]
}
