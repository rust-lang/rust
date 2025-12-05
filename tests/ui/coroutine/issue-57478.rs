//@ check-pass

#![feature(negative_impls, coroutines, stmt_expr_attributes)]

struct Foo;
impl !Send for Foo {}

fn main() {
    assert_send(
        #[coroutine]
        || {
            let guard = Foo;
            drop(guard);
            yield;
        },
    )
}

fn assert_send<T: Send>(_: T) {}
