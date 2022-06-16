use crate::{
    cell::LazyCell,
    lazy::{SyncLazy, SyncOnceCell},
    panic,
    sync::{
        atomic::{AtomicUsize, Ordering::SeqCst},
        mpsc::channel,
        Mutex,
    },
    thread,
};

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

fn spawn_and_wait<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    thread::spawn(f).join().unwrap()
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn sync_once_cell() {
    static ONCE_CELL: SyncOnceCell<i32> = SyncOnceCell::new();

    assert!(ONCE_CELL.get().is_none());

    spawn_and_wait(|| {
        ONCE_CELL.get_or_init(|| 92);
        assert_eq!(ONCE_CELL.get(), Some(&92));
    });

    ONCE_CELL.get_or_init(|| panic!("Kabom!"));
    assert_eq!(ONCE_CELL.get(), Some(&92));
}

#[test]
fn sync_once_cell_get_mut() {
    let mut c = SyncOnceCell::new();
    assert!(c.get_mut().is_none());
    c.set(90).unwrap();
    *c.get_mut().unwrap() += 2;
    assert_eq!(c.get_mut(), Some(&mut 92));
}

#[test]
fn sync_once_cell_get_unchecked() {
    let c = SyncOnceCell::new();
    c.set(92).unwrap();
    unsafe {
        assert_eq!(c.get_unchecked(), &92);
    }
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn sync_once_cell_drop() {
    static DROP_CNT: AtomicUsize = AtomicUsize::new(0);
    struct Dropper;
    impl Drop for Dropper {
        fn drop(&mut self) {
            DROP_CNT.fetch_add(1, SeqCst);
        }
    }

    let x = SyncOnceCell::new();
    spawn_and_wait(move || {
        x.get_or_init(|| Dropper);
        assert_eq!(DROP_CNT.load(SeqCst), 0);
        drop(x);
    });

    assert_eq!(DROP_CNT.load(SeqCst), 1);
}

#[test]
fn sync_once_cell_drop_empty() {
    let x = SyncOnceCell::<String>::new();
    drop(x);
}

#[test]
fn clone() {
    let s = SyncOnceCell::new();
    let c = s.clone();
    assert!(c.get().is_none());

    s.set("hello".to_string()).unwrap();
    let c = s.clone();
    assert_eq!(c.get().map(String::as_str), Some("hello"));
}

#[test]
fn get_or_try_init() {
    let cell: SyncOnceCell<String> = SyncOnceCell::new();
    assert!(cell.get().is_none());

    let res = panic::catch_unwind(|| cell.get_or_try_init(|| -> Result<_, ()> { panic!() }));
    assert!(res.is_err());
    assert!(!cell.is_initialized());
    assert!(cell.get().is_none());

    assert_eq!(cell.get_or_try_init(|| Err(())), Err(()));

    assert_eq!(cell.get_or_try_init(|| Ok::<_, ()>("hello".to_string())), Ok(&"hello".to_string()));
    assert_eq!(cell.get(), Some(&"hello".to_string()));
}

#[test]
fn from_impl() {
    assert_eq!(SyncOnceCell::from("value").get(), Some(&"value"));
    assert_ne!(SyncOnceCell::from("foo").get(), Some(&"bar"));
}

#[test]
fn partialeq_impl() {
    assert!(SyncOnceCell::from("value") == SyncOnceCell::from("value"));
    assert!(SyncOnceCell::from("foo") != SyncOnceCell::from("bar"));

    assert!(SyncOnceCell::<String>::new() == SyncOnceCell::new());
    assert!(SyncOnceCell::<String>::new() != SyncOnceCell::from("value".to_owned()));
}

#[test]
fn into_inner() {
    let cell: SyncOnceCell<String> = SyncOnceCell::new();
    assert_eq!(cell.into_inner(), None);
    let cell = SyncOnceCell::new();
    cell.set("hello".to_string()).unwrap();
    assert_eq!(cell.into_inner(), Some("hello".to_string()));
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn sync_lazy_new() {
    static CALLED: AtomicUsize = AtomicUsize::new(0);
    static SYNC_LAZY: SyncLazy<i32> = SyncLazy::new(|| {
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

    let lazy: SyncLazy<Mutex<Foo>> = <_>::default();

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
    static XS: SyncLazy<Vec<i32>> = SyncLazy::new(|| {
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
        static XS: SyncOnceCell<Vec<i32>> = SyncOnceCell::new();
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
    let x: SyncLazy<String> = SyncLazy::new(|| panic!("kaboom"));
    for _ in 0..2 {
        let res = panic::catch_unwind(|| x.len());
        assert!(res.is_err());
    }
}

#[test]
fn is_sync_send() {
    fn assert_traits<T: Send + Sync>() {}
    assert_traits::<SyncOnceCell<String>>();
    assert_traits::<SyncLazy<String>>();
}

#[test]
fn eval_once_macro() {
    macro_rules! eval_once {
        (|| -> $ty:ty {
            $($body:tt)*
        }) => {{
            static ONCE_CELL: SyncOnceCell<$ty> = SyncOnceCell::new();
            fn init() -> $ty {
                $($body)*
            }
            ONCE_CELL.get_or_init(init)
        }};
    }

    let fib: &'static Vec<i32> = eval_once! {
        || -> Vec<i32> {
            let mut res = vec![1, 1];
            for i in 0..10 {
                let next = res[i] + res[i + 1];
                res.push(next);
            }
            res
        }
    };
    assert_eq!(fib[5], 8)
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn sync_once_cell_does_not_leak_partially_constructed_boxes() {
    static ONCE_CELL: SyncOnceCell<String> = SyncOnceCell::new();

    let n_readers = 10;
    let n_writers = 3;
    const MSG: &str = "Hello, World";

    let (tx, rx) = channel();

    for _ in 0..n_readers {
        let tx = tx.clone();
        thread::spawn(move || {
            loop {
                if let Some(msg) = ONCE_CELL.get() {
                    tx.send(msg).unwrap();
                    break;
                }
                #[cfg(target_env = "sgx")]
                crate::thread::yield_now();
            }
        });
    }
    for _ in 0..n_writers {
        thread::spawn(move || {
            let _ = ONCE_CELL.set(MSG.to_owned());
        });
    }

    for _ in 0..n_readers {
        let msg = rx.recv().unwrap();
        assert_eq!(msg, MSG);
    }
}

#[test]
fn dropck() {
    let cell = SyncOnceCell::new();
    {
        let s = String::new();
        cell.set(&s).unwrap();
    }
}
