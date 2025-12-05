//@ run-pass

//@ revisions: default nomiropt
//@[nomiropt]compile-flags: -Z mir-opt-level=0

//@ needs-threads
//@ compile-flags: --test

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{CoroutineState, Coroutine};
use std::pin::Pin;
use std::thread;

#[test]
fn simple() {
    let mut foo = #[coroutine] || {
        if false {
            yield;
        }
    };

    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn return_capture() {
    let a = String::from("foo");
    let mut foo = #[coroutine] || {
        if false {
            yield;
        }
        a
    };

    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Complete(ref s) if *s == "foo" => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn simple_yield() {
    let mut foo = #[coroutine] || {
        yield;
    };

    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Yielded(()) => {}
        s => panic!("bad state: {:?}", s),
    }
    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn yield_capture() {
    let b = String::from("foo");
    let mut foo = #[coroutine] || {
        yield b;
    };

    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Yielded(ref s) if *s == "foo" => {}
        s => panic!("bad state: {:?}", s),
    }
    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn simple_yield_value() {
    let mut foo = #[coroutine] || {
        yield String::from("bar");
        return String::from("foo")
    };

    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Yielded(ref s) if *s == "bar" => {}
        s => panic!("bad state: {:?}", s),
    }
    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Complete(ref s) if *s == "foo" => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn return_after_yield() {
    let a = String::from("foo");
    let mut foo = #[coroutine] || {
        yield;
        return a
    };

    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Yielded(()) => {}
        s => panic!("bad state: {:?}", s),
    }
    match Pin::new(&mut foo).resume(()) {
        CoroutineState::Complete(ref s) if *s == "foo" => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn send_and_sync() {
    assert_send_sync(#[coroutine] || {
        yield
    });
    assert_send_sync(#[coroutine] || {
        yield String::from("foo");
    });
    assert_send_sync(#[coroutine] || {
        yield;
        return String::from("foo");
    });
    let a = 3;
    assert_send_sync(#[coroutine] || {
        yield a;
        return
    });
    let a = 3;
    assert_send_sync(#[coroutine] move || {
        yield a;
        return
    });
    let a = String::from("a");
    assert_send_sync(#[coroutine] || {
        yield ;
        drop(a);
        return
    });
    let a = String::from("a");
    assert_send_sync(#[coroutine] move || {
        yield ;
        drop(a);
        return
    });

    fn assert_send_sync<T: Send + Sync>(_: T) {}
}

#[test]
fn send_over_threads() {
    let mut foo = #[coroutine] || { yield };
    thread::spawn(move || {
        match Pin::new(&mut foo).resume(()) {
            CoroutineState::Yielded(()) => {}
            s => panic!("bad state: {:?}", s),
        }
        match Pin::new(&mut foo).resume(()) {
            CoroutineState::Complete(()) => {}
            s => panic!("bad state: {:?}", s),
        }
    }).join().unwrap();

    let a = String::from("a");
    let mut foo = #[coroutine] || { yield a };
    thread::spawn(move || {
        match Pin::new(&mut foo).resume(()) {
            CoroutineState::Yielded(ref s) if *s == "a" => {}
            s => panic!("bad state: {:?}", s),
        }
        match Pin::new(&mut foo).resume(()) {
            CoroutineState::Complete(()) => {}
            s => panic!("bad state: {:?}", s),
        }
    }).join().unwrap();
}
