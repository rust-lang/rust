use crate::{
    cell::LazyCell,
    panic,
    sync::{
        atomic::{AtomicUsize, Ordering::SeqCst},
        Mutex,
    },
    sync::{LazyLock, OnceLock},
    thread,
};

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
fn lazy_poisoning() {
    let x: LazyCell<String> = LazyCell::new(|| panic!("kaboom"));
    for _ in 0..2 {
        let res = panic::catch_unwind(panic::AssertUnwindSafe(|| x.len()));
        assert!(res.is_err());
    }
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
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
#[cfg_attr(target_os = "emscripten", ignore)]
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

#[test]
fn sync_lazy_poisoning() {
    let x: LazyLock<String> = LazyLock::new(|| panic!("kaboom"));
    for _ in 0..2 {
        let res = panic::catch_unwind(|| x.len());
        assert!(res.is_err());
    }
}

#[test]
fn is_sync_send() {
    fn assert_traits<T: Send + Sync>() {}
    assert_traits::<LazyLock<String>>();
}
