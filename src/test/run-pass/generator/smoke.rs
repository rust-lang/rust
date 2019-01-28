// run-pass

// ignore-emscripten no threads support
// compile-flags: --test

#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::pin::Pin;
use std::thread;

#[test]
fn simple() {
    let mut foo = || {
        if false {
            yield;
        }
    };

    match Pin::new(&mut foo).resume() {
        GeneratorState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn return_capture() {
    let a = String::from("foo");
    let mut foo = || {
        if false {
            yield;
        }
        a
    };

    match Pin::new(&mut foo).resume() {
        GeneratorState::Complete(ref s) if *s == "foo" => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn simple_yield() {
    let mut foo = || {
        yield;
    };

    match Pin::new(&mut foo).resume() {
        GeneratorState::Yielded(()) => {}
        s => panic!("bad state: {:?}", s),
    }
    match Pin::new(&mut foo).resume() {
        GeneratorState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn yield_capture() {
    let b = String::from("foo");
    let mut foo = || {
        yield b;
    };

    match Pin::new(&mut foo).resume() {
        GeneratorState::Yielded(ref s) if *s == "foo" => {}
        s => panic!("bad state: {:?}", s),
    }
    match Pin::new(&mut foo).resume() {
        GeneratorState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn simple_yield_value() {
    let mut foo = || {
        yield String::from("bar");
        return String::from("foo")
    };

    match Pin::new(&mut foo).resume() {
        GeneratorState::Yielded(ref s) if *s == "bar" => {}
        s => panic!("bad state: {:?}", s),
    }
    match Pin::new(&mut foo).resume() {
        GeneratorState::Complete(ref s) if *s == "foo" => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn return_after_yield() {
    let a = String::from("foo");
    let mut foo = || {
        yield;
        return a
    };

    match Pin::new(&mut foo).resume() {
        GeneratorState::Yielded(()) => {}
        s => panic!("bad state: {:?}", s),
    }
    match Pin::new(&mut foo).resume() {
        GeneratorState::Complete(ref s) if *s == "foo" => {}
        s => panic!("bad state: {:?}", s),
    }
}

#[test]
fn send_and_sync() {
    assert_send_sync(|| {
        yield
    });
    assert_send_sync(|| {
        yield String::from("foo");
    });
    assert_send_sync(|| {
        yield;
        return String::from("foo");
    });
    let a = 3;
    assert_send_sync(|| {
        yield a;
        return
    });
    let a = 3;
    assert_send_sync(move || {
        yield a;
        return
    });
    let a = String::from("a");
    assert_send_sync(|| {
        yield ;
        drop(a);
        return
    });
    let a = String::from("a");
    assert_send_sync(move || {
        yield ;
        drop(a);
        return
    });

    fn assert_send_sync<T: Send + Sync>(_: T) {}
}

#[test]
fn send_over_threads() {
    let mut foo = || { yield };
    thread::spawn(move || {
        match Pin::new(&mut foo).resume() {
            GeneratorState::Yielded(()) => {}
            s => panic!("bad state: {:?}", s),
        }
        match Pin::new(&mut foo).resume() {
            GeneratorState::Complete(()) => {}
            s => panic!("bad state: {:?}", s),
        }
    }).join().unwrap();

    let a = String::from("a");
    let mut foo = || { yield a };
    thread::spawn(move || {
        match Pin::new(&mut foo).resume() {
            GeneratorState::Yielded(ref s) if *s == "a" => {}
            s => panic!("bad state: {:?}", s),
        }
        match Pin::new(&mut foo).resume() {
            GeneratorState::Complete(()) => {}
            s => panic!("bad state: {:?}", s),
        }
    }).join().unwrap();
}
