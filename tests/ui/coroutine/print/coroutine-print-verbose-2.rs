//@ compile-flags: -Zverbose-internals

// Same as test/ui/coroutine/not-send-sync.rs
#![feature(coroutines, stmt_expr_attributes)]
#![feature(negative_impls)]

struct NotSend;
struct NotSync;

impl !Send for NotSend {}
impl !Sync for NotSync {}

fn main() {
    fn assert_sync<T: Sync>(_: T) {}
    fn assert_send<T: Send>(_: T) {}

    assert_sync(#[coroutine] || {
        //~^ ERROR: coroutine cannot be shared between threads safely
        let a = NotSync;
        yield;
        drop(a);
    });

    assert_send(#[coroutine] || {
        //~^ ERROR: coroutine cannot be sent between threads safely
        let a = NotSend;
        yield;
        drop(a);
    });
}
