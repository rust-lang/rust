//@ run-pass

#![feature(coroutines, coroutine_trait)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn foo(_: &str) -> String {
    String::new()
}

fn bar(baz: String) -> impl Coroutine<(), Yield = String, Return = ()> {
    #[coroutine] move || {
        yield foo(&baz);
    }
}

fn foo2(_: &str) -> Result<String, ()> {
    Err(())
}

fn bar2(baz: String) -> impl Coroutine<(), Yield = String, Return = ()> {
    #[coroutine] move || {
        if let Ok(quux) = foo2(&baz) {
            yield quux;
        }
    }
}

fn main() {
    assert_eq!(
        Pin::new(&mut bar(String::new())).resume(()),
        CoroutineState::Yielded(String::new())
    );
    assert_eq!(Pin::new(&mut bar2(String::new())).resume(()), CoroutineState::Complete(()));
}
