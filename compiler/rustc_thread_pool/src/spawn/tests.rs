use std::any::Any;
use std::sync::Mutex;
use std::sync::mpsc::channel;

use super::{spawn, spawn_fifo};
use crate::{ThreadPoolBuilder, scope};

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn spawn_then_join_in_worker() {
    let (tx, rx) = channel();
    scope(move |_| {
        spawn(move || tx.send(22).unwrap());
    });
    assert_eq!(22, rx.recv().unwrap());
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn spawn_then_join_outside_worker() {
    let (tx, rx) = channel();
    spawn(move || tx.send(22).unwrap());
    assert_eq!(22, rx.recv().unwrap());
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore)]
fn panic_fwd() {
    let (tx, rx) = channel();

    let tx = Mutex::new(tx);
    let panic_handler = move |err: Box<dyn Any + Send>| {
        let tx = tx.lock().unwrap();
        if let Some(&msg) = err.downcast_ref::<&str>() {
            if msg == "Hello, world!" {
                tx.send(1).unwrap();
            } else {
                tx.send(2).unwrap();
            }
        } else {
            tx.send(3).unwrap();
        }
    };

    let builder = ThreadPoolBuilder::new().panic_handler(panic_handler);

    builder.build().unwrap().spawn(move || panic!("Hello, world!"));

    assert_eq!(1, rx.recv().unwrap());
}

/// Test what happens when the thread-pool is dropped but there are
/// still active asynchronous tasks. We expect the thread-pool to stay
/// alive and executing until those threads are complete.
#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn termination_while_things_are_executing() {
    let (tx0, rx0) = channel();
    let (tx1, rx1) = channel();

    // Create a thread-pool and spawn some code in it, but then drop
    // our reference to it.
    {
        let thread_pool = ThreadPoolBuilder::new().build().unwrap();
        thread_pool.spawn(move || {
            let data = rx0.recv().unwrap();

            // At this point, we know the "main" reference to the
            // `ThreadPool` has been dropped, but there are still
            // active threads. Launch one more.
            spawn(move || {
                tx1.send(data).unwrap();
            });
        });
    }

    tx0.send(22).unwrap();
    let v = rx1.recv().unwrap();
    assert_eq!(v, 22);
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore)]
fn custom_panic_handler_and_spawn() {
    let (tx, rx) = channel();

    // Create a parallel closure that will send panics on the
    // channel; since the closure is potentially executed in parallel
    // with itself, we have to wrap `tx` in a mutex.
    let tx = Mutex::new(tx);
    let panic_handler = move |e: Box<dyn Any + Send>| {
        tx.lock().unwrap().send(e).unwrap();
    };

    // Execute an async that will panic.
    let builder = ThreadPoolBuilder::new().panic_handler(panic_handler);
    builder.build().unwrap().spawn(move || {
        panic!("Hello, world!");
    });

    // Check that we got back the panic we expected.
    let error = rx.recv().unwrap();
    if let Some(&msg) = error.downcast_ref::<&str>() {
        assert_eq!(msg, "Hello, world!");
    } else {
        panic!("did not receive a string from panic handler");
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore)]
fn custom_panic_handler_and_nested_spawn() {
    let (tx, rx) = channel();

    // Create a parallel closure that will send panics on the
    // channel; since the closure is potentially executed in parallel
    // with itself, we have to wrap `tx` in a mutex.
    let tx = Mutex::new(tx);
    let panic_handler = move |e| {
        tx.lock().unwrap().send(e).unwrap();
    };

    // Execute an async that will (eventually) panic.
    const PANICS: usize = 3;
    let builder = ThreadPoolBuilder::new().panic_handler(panic_handler);
    builder.build().unwrap().spawn(move || {
        // launch 3 nested spawn-asyncs; these should be in the same
        // thread-pool and hence inherit the same panic handler
        for _ in 0..PANICS {
            spawn(move || {
                panic!("Hello, world!");
            });
        }
    });

    // Check that we get back the panics we expected.
    for _ in 0..PANICS {
        let error = rx.recv().unwrap();
        if let Some(&msg) = error.downcast_ref::<&str>() {
            assert_eq!(msg, "Hello, world!");
        } else {
            panic!("did not receive a string from panic handler");
        }
    }
}

macro_rules! test_order {
    ($outer_spawn:ident, $inner_spawn:ident) => {{
        let builder = ThreadPoolBuilder::new().num_threads(1);
        let pool = builder.build().unwrap();
        let (tx, rx) = channel();
        pool.install(move || {
            for i in 0..10 {
                let tx = tx.clone();
                $outer_spawn(move || {
                    for j in 0..10 {
                        let tx = tx.clone();
                        $inner_spawn(move || {
                            tx.send(i * 10 + j).unwrap();
                        });
                    }
                });
            }
        });
        rx.iter().collect::<Vec<i32>>()
    }};
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn lifo_order() {
    // In the absence of stealing, `spawn()` jobs on a thread will run in LIFO order.
    let vec = test_order!(spawn, spawn);
    let expected: Vec<i32> = (0..100).rev().collect(); // LIFO -> reversed
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn fifo_order() {
    // In the absence of stealing, `spawn_fifo()` jobs on a thread will run in FIFO order.
    let vec = test_order!(spawn_fifo, spawn_fifo);
    let expected: Vec<i32> = (0..100).collect(); // FIFO -> natural order
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn lifo_fifo_order() {
    // LIFO on the outside, FIFO on the inside
    let vec = test_order!(spawn, spawn_fifo);
    let expected: Vec<i32> = (0..10).rev().flat_map(|i| (0..10).map(move |j| i * 10 + j)).collect();
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn fifo_lifo_order() {
    // FIFO on the outside, LIFO on the inside
    let vec = test_order!(spawn_fifo, spawn);
    let expected: Vec<i32> = (0..10).flat_map(|i| (0..10).rev().map(move |j| i * 10 + j)).collect();
    assert_eq!(vec, expected);
}

macro_rules! spawn_send {
    ($spawn:ident, $tx:ident, $i:expr) => {{
        let tx = $tx.clone();
        $spawn(move || tx.send($i).unwrap());
    }};
}

/// Test mixed spawns pushing a series of numbers, interleaved such
/// such that negative values are using the second kind of spawn.
macro_rules! test_mixed_order {
    ($pos_spawn:ident, $neg_spawn:ident) => {{
        let builder = ThreadPoolBuilder::new().num_threads(1);
        let pool = builder.build().unwrap();
        let (tx, rx) = channel();
        pool.install(move || {
            spawn_send!($pos_spawn, tx, 0);
            spawn_send!($neg_spawn, tx, -1);
            spawn_send!($pos_spawn, tx, 1);
            spawn_send!($neg_spawn, tx, -2);
            spawn_send!($pos_spawn, tx, 2);
            spawn_send!($neg_spawn, tx, -3);
            spawn_send!($pos_spawn, tx, 3);
        });
        rx.iter().collect::<Vec<i32>>()
    }};
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn mixed_lifo_fifo_order() {
    let vec = test_mixed_order!(spawn, spawn_fifo);
    let expected = vec![3, -1, 2, -2, 1, -3, 0];
    assert_eq!(vec, expected);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn mixed_fifo_lifo_order() {
    let vec = test_mixed_order!(spawn_fifo, spawn);
    let expected = vec![0, -3, 1, -2, 2, -1, 3];
    assert_eq!(vec, expected);
}
