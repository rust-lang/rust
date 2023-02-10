#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::{Debug, Display};

fn main() {
    make_dyn_star();
    method();
    box_();
    dispatch_on_pin_mut();
    dyn_star_to_dyn();
    dyn_to_dyn_star();
}

fn dyn_star_to_dyn() {
    let x: dyn* Debug = &42;
    let x = Box::new(x) as Box<dyn Debug>;
    assert_eq!("42", format!("{x:?}"));
}

fn dyn_to_dyn_star() {
    let x: Box<dyn Debug> = Box::new(42);
    let x = &x as dyn* Debug;
    assert_eq!("42", format!("{x:?}"));
}

fn make_dyn_star() {
    fn make_dyn_star_coercion(i: usize) {
        let _dyn_i: dyn* Debug = i;
    }

    fn make_dyn_star_explicit(i: usize) {
        let _dyn_i: dyn* Debug = i as dyn* Debug;
    }

    make_dyn_star_coercion(42);
    make_dyn_star_explicit(42);
}

fn method() {
    trait Foo {
        fn get(&self) -> usize;
    }
    
    impl Foo for usize {
        fn get(&self) -> usize {
            *self
        }
    }
    
    fn invoke_dyn_star(i: dyn* Foo) -> usize {
        i.get()
    }
    
    fn make_and_invoke_dyn_star(i: usize) -> usize {
        let dyn_i: dyn* Foo = i;
        invoke_dyn_star(dyn_i)
    }
    
    assert_eq!(make_and_invoke_dyn_star(42), 42);
}

fn box_() {
    fn make_dyn_star() -> dyn* Display {
        Box::new(42) as dyn* Display
    }
    
    let x = make_dyn_star();
    assert_eq!(format!("{x}"), "42");
}

fn dispatch_on_pin_mut() {
    use std::future::Future;

    async fn foo(f: dyn* Future<Output = i32>) {
        println!("dispatch_on_pin_mut: value: {}", f.await);
    }

    async fn async_main() {
        foo(Box::pin(async { 1 })).await
    }

    // ------------------------------------------------------------------------- //
    // Implementation Details Below...

    use std::pin::Pin;
    use std::task::*;

    pub fn noop_waker() -> Waker {
        let raw = RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE);

        // SAFETY: the contracts for RawWaker and RawWakerVTable are upheld
        unsafe { Waker::from_raw(raw) }
    }

    const NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

    unsafe fn noop_clone(_p: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE)
    }

    unsafe fn noop(_p: *const ()) {}

    let mut fut = async_main();

    // Poll loop, just to test the future...
    let waker = noop_waker();
    let ctx = &mut Context::from_waker(&waker);

    loop {
        match unsafe { Pin::new_unchecked(&mut fut).poll(ctx) } {
            Poll::Pending => {}
            Poll::Ready(()) => break,
        }
    }
}
