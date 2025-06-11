#![cfg(test)]

use crate::{ThreadPoolBuildError, ThreadPoolBuilder};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn worker_thread_index() {
    let pool = ThreadPoolBuilder::new().num_threads(22).build().unwrap();
    assert_eq!(pool.current_num_threads(), 22);
    assert_eq!(pool.current_thread_index(), None);
    let index = pool.install(|| pool.current_thread_index().unwrap());
    assert!(index < 22);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn start_callback_called() {
    let n_threads = 16;
    let n_called = Arc::new(AtomicUsize::new(0));
    // Wait for all the threads in the pool plus the one running tests.
    let barrier = Arc::new(Barrier::new(n_threads + 1));

    let b = Arc::clone(&barrier);
    let nc = Arc::clone(&n_called);
    let start_handler = move |_| {
        nc.fetch_add(1, Ordering::SeqCst);
        b.wait();
    };

    let conf = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .start_handler(start_handler);
    let _ = conf.build().unwrap();

    // Wait for all the threads to have been scheduled to run.
    barrier.wait();

    // The handler must have been called on every started thread.
    assert_eq!(n_called.load(Ordering::SeqCst), n_threads);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn exit_callback_called() {
    let n_threads = 16;
    let n_called = Arc::new(AtomicUsize::new(0));
    // Wait for all the threads in the pool plus the one running tests.
    let barrier = Arc::new(Barrier::new(n_threads + 1));

    let b = Arc::clone(&barrier);
    let nc = Arc::clone(&n_called);
    let exit_handler = move |_| {
        nc.fetch_add(1, Ordering::SeqCst);
        b.wait();
    };

    let conf = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .exit_handler(exit_handler);
    {
        let _ = conf.build().unwrap();
        // Drop the pool so it stops the running threads.
    }

    // Wait for all the threads to have been scheduled to run.
    barrier.wait();

    // The handler must have been called on every exiting thread.
    assert_eq!(n_called.load(Ordering::SeqCst), n_threads);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn handler_panics_handled_correctly() {
    let n_threads = 16;
    let n_called = Arc::new(AtomicUsize::new(0));
    // Wait for all the threads in the pool plus the one running tests.
    let start_barrier = Arc::new(Barrier::new(n_threads + 1));
    let exit_barrier = Arc::new(Barrier::new(n_threads + 1));

    let start_handler = move |_| {
        panic!("ensure panic handler is called when starting");
    };
    let exit_handler = move |_| {
        panic!("ensure panic handler is called when exiting");
    };

    let sb = Arc::clone(&start_barrier);
    let eb = Arc::clone(&exit_barrier);
    let nc = Arc::clone(&n_called);
    let panic_handler = move |_| {
        let val = nc.fetch_add(1, Ordering::SeqCst);
        if val < n_threads {
            sb.wait();
        } else {
            eb.wait();
        }
    };

    let conf = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .start_handler(start_handler)
        .exit_handler(exit_handler)
        .panic_handler(panic_handler);
    {
        let _ = conf.build().unwrap();

        // Wait for all the threads to start, panic in the start handler,
        // and been taken care of by the panic handler.
        start_barrier.wait();

        // Drop the pool so it stops the running threads.
    }

    // Wait for all the threads to exit, panic in the exit handler,
    // and been taken care of by the panic handler.
    exit_barrier.wait();

    // The panic handler must have been called twice on every thread.
    assert_eq!(n_called.load(Ordering::SeqCst), 2 * n_threads);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn check_config_build() {
    let pool = ThreadPoolBuilder::new().num_threads(22).build().unwrap();
    assert_eq!(pool.current_num_threads(), 22);
}

/// Helper used by check_error_send_sync to ensure ThreadPoolBuildError is Send + Sync
fn _send_sync<T: Send + Sync>() {}

#[test]
fn check_error_send_sync() {
    _send_sync::<ThreadPoolBuildError>();
}

#[allow(deprecated)]
#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn configuration() {
    let start_handler = move |_| {};
    let exit_handler = move |_| {};
    let panic_handler = move |_| {};
    let thread_name = move |i| format!("thread_name_{}", i);

    // Ensure we can call all public methods on Configuration
    crate::Configuration::new()
        .thread_name(thread_name)
        .num_threads(5)
        .panic_handler(panic_handler)
        .stack_size(4e6 as usize)
        .breadth_first()
        .start_handler(start_handler)
        .exit_handler(exit_handler)
        .build()
        .unwrap();
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn default_pool() {
    ThreadPoolBuilder::default().build().unwrap();
}

/// Test that custom spawned threads get their `WorkerThread` cleared once
/// the pool is done with them, allowing them to be used with rayon again
/// later. e.g. WebAssembly want to have their own pool of available threads.
#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn cleared_current_thread() -> Result<(), ThreadPoolBuildError> {
    let n_threads = 5;
    let mut handles = vec![];
    let pool = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .spawn_handler(|thread| {
            let handle = std::thread::spawn(move || {
                thread.run();

                // Afterward, the current thread shouldn't be set anymore.
                assert_eq!(crate::current_thread_index(), None);
            });
            handles.push(handle);
            Ok(())
        })
        .build()?;
    assert_eq!(handles.len(), n_threads);

    pool.install(|| assert!(crate::current_thread_index().is_some()));
    drop(pool);

    // Wait for all threads to make their assertions and exit
    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}
