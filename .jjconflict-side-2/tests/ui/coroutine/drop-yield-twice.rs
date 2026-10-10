#![feature(negative_impls, coroutines, stmt_expr_attributes)]

struct Foo(i32);
impl !Send for Foo {}

fn main() {
    assert_send(#[coroutine] || { //~ ERROR coroutine cannot be sent between threads safely
        let guard = Foo(42);
        yield;
        drop(guard);
        yield;
    })
}

fn assert_send<T: Send>(_: T) {}
