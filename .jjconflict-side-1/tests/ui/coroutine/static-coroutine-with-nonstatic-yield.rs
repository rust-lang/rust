//@ check-pass
//@ known-bug: #144442

// Same family as #84366 / #112905: a coroutine that yields a non-`'static`
// reference is wrongly `: 'static`, allowing a `Box<dyn Any>` downcast to
// transmute between distinct lifetime substitutions and produce a UAF.

#![forbid(unsafe_code)] // No `unsafe!`
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::any::Any;
use std::cell::RefCell;
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;
use std::rc::Rc;

type Payload = Box<i32>;

fn make_coro<'a>()
-> impl Coroutine<Yield = Rc<RefCell<Option<&'a Payload>>>, Return = ()> + 'static {
    #[coroutine]
    || {
        let storage: Rc<RefCell<Option<&'a Payload>>> = Rc::new(RefCell::new(None));
        yield storage.clone();
        yield storage;
    }
}

pub fn expand<'a>(payload: &'a Payload) -> &'static Payload {
    let mut coro1 = Box::pin(make_coro::<'a>());
    let coro2 = make_coro::<'static>();
    let CoroutineState::Yielded(storage) = coro1.as_mut().resume(()) else {
        panic!()
    };
    *storage.borrow_mut() = Some(payload);
    extract(coro1, coro2)
}

fn extract<
    'a,
    F: Coroutine<Yield = Rc<RefCell<Option<&'a Payload>>>, Return = ()> + 'static,
    G: Coroutine<Yield = Rc<RefCell<Option<&'static Payload>>>, Return = ()> + 'static,
>(
    x: Pin<Box<F>>,
    _: G,
) -> &'static Payload {
    let mut g: Pin<Box<G>> = *(Box::new(x) as Box<dyn Any>).downcast().unwrap();
    let CoroutineState::Yielded(storage) = g.as_mut().resume(()) else {
        panic!()
    };
    let payload = storage.borrow().unwrap();
    payload
}

fn main() {
    let x = Box::new(Box::new(1i32));
    let y = expand(&x);
    drop(x);
    println!("{y}"); // Segfaults — UAF without `unsafe`.
}
