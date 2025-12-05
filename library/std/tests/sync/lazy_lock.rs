use std::cell::LazyCell;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::{LazyLock, Mutex, OnceLock};
use std::{panic, thread};

fn spawn_and_wait<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    thread::spawn(f).join().unwrap()
}

#[test]
fn lazy_default() {
    static CALLED: AtomicUsize = AtomicUsize::new(0);

    struct Foo(u8);
    impl Default for Foo {
        fn default() -> Self {
            CALLED.fetch_add(1, SeqCst);
            Foo(42)
        }
    }

    let lazy: LazyCell<Mutex<Foo>> = <_>::default();

    assert_eq!(CALLED.load(SeqCst), 0);

    assert_eq!(lazy.lock().unwrap().0, 42);
    assert_eq!(CALLED.load(SeqCst), 1);

    lazy.lock().unwrap().0 = 21;

    assert_eq!(lazy.lock().unwrap().0, 21);
    assert_eq!(CALLED.load(SeqCst), 1);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn sync_lazy_new() {
    static CALLED: AtomicUsize = AtomicUsize::new(0);
    static SYNC_LAZY: LazyLock<i32> = LazyLock::new(|| {
        CALLED.fetch_add(1, SeqCst);
        92
    });

    assert_eq!(CALLED.load(SeqCst), 0);

    spawn_and_wait(|| {
        let y = *SYNC_LAZY - 30;
        assert_eq!(y, 62);
        assert_eq!(CALLED.load(SeqCst), 1);
    });

    let y = *SYNC_LAZY - 30;
    assert_eq!(y, 62);
    assert_eq!(CALLED.load(SeqCst), 1);
}

#[test]
fn sync_lazy_default() {
    static CALLED: AtomicUsize = AtomicUsize::new(0);

    struct Foo(u8);
    impl Default for Foo {
        fn default() -> Self {
            CALLED.fetch_add(1, SeqCst);
            Foo(42)
        }
    }

    let lazy: LazyLock<Mutex<Foo>> = <_>::default();

    assert_eq!(CALLED.load(SeqCst), 0);

    assert_eq!(lazy.lock().unwrap().0, 42);
    assert_eq!(CALLED.load(SeqCst), 1);

    lazy.lock().unwrap().0 = 21;

    assert_eq!(lazy.lock().unwrap().0, 21);
    assert_eq!(CALLED.load(SeqCst), 1);
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn static_sync_lazy() {
    static XS: LazyLock<Vec<i32>> = LazyLock::new(|| {
        let mut xs = Vec::new();
        xs.push(1);
        xs.push(2);
        xs.push(3);
        xs
    });

    spawn_and_wait(|| {
        assert_eq!(&*XS, &vec![1, 2, 3]);
    });

    assert_eq!(&*XS, &vec![1, 2, 3]);
}

#[test]
fn static_sync_lazy_via_fn() {
    fn xs() -> &'static Vec<i32> {
        static XS: OnceLock<Vec<i32>> = OnceLock::new();
        XS.get_or_init(|| {
            let mut xs = Vec::new();
            xs.push(1);
            xs.push(2);
            xs.push(3);
            xs
        })
    }
    assert_eq!(xs(), &vec![1, 2, 3]);
}

// Check that we can infer `T` from closure's type.
#[test]
fn lazy_type_inference() {
    let _ = LazyCell::new(|| ());
}

#[test]
fn is_sync_send() {
    fn assert_traits<T: Send + Sync>() {}
    assert_traits::<LazyLock<String>>();
}

#[test]
fn lazy_force_mut() {
    let s = "abc".to_owned();
    let mut lazy = LazyLock::new(move || s);
    LazyLock::force_mut(&mut lazy);
    let p = LazyLock::force_mut(&mut lazy);
    p.clear();
    LazyLock::force_mut(&mut lazy);
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn lazy_poisoning() {
    let x: LazyCell<String> = LazyCell::new(|| panic!("kaboom"));
    for _ in 0..2 {
        let res = panic::catch_unwind(panic::AssertUnwindSafe(|| x.len()));
        assert!(res.is_err());
    }
}

/// Verifies that when a `LazyLock` is poisoned, it panics with the correct error message ("LazyLock
/// instance has previously been poisoned") instead of the underlying `Once` error message.
#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
#[should_panic(expected = "LazyLock instance has previously been poisoned")]
fn lazy_lock_deref_panic() {
    let lazy: LazyLock<String> = LazyLock::new(|| panic!("initialization failed"));

    // First access will panic during initialization.
    let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _ = &*lazy;
    }));

    // Second access should panic with the poisoned message.
    let _ = &*lazy;
}

#[test]
#[should_panic(expected = "LazyLock instance has previously been poisoned")]
fn lazy_lock_deref_mut_panic() {
    let mut lazy: LazyLock<String> = LazyLock::new(|| panic!("initialization failed"));

    // First access will panic during initialization.
    let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _ = LazyLock::force_mut(&mut lazy);
    }));

    // Second access should panic with the poisoned message.
    let _ = &*lazy;
}

/// Verifies that when the initialization closure panics with a custom message, that message is
/// preserved and not overridden by `LazyLock`.
#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
#[should_panic(expected = "custom panic message from closure")]
fn lazy_lock_preserves_closure_panic_message() {
    let lazy: LazyLock<String> = LazyLock::new(|| panic!("custom panic message from closure"));

    // This should panic with the original message from the closure.
    let _ = &*lazy;
}
