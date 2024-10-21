use crate::sync::OnceLock;
use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering::SeqCst;
use crate::sync::mpsc::channel;
use crate::{panic, thread};

fn spawn_and_wait<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    thread::spawn(f).join().unwrap()
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn sync_once_cell() {
    static ONCE_CELL: OnceLock<i32> = OnceLock::new();

    assert!(ONCE_CELL.get().is_none());

    spawn_and_wait(|| {
        ONCE_CELL.get_or_init(|| 92);
        assert_eq!(ONCE_CELL.get(), Some(&92));
    });

    ONCE_CELL.get_or_init(|| panic!("Kaboom!"));
    assert_eq!(ONCE_CELL.get(), Some(&92));
}

#[test]
fn sync_once_cell_get_mut() {
    let mut c = OnceLock::new();
    assert!(c.get_mut().is_none());
    c.set(90).unwrap();
    *c.get_mut().unwrap() += 2;
    assert_eq!(c.get_mut(), Some(&mut 92));
}

#[test]
fn sync_once_cell_get_unchecked() {
    let c = OnceLock::new();
    c.set(92).unwrap();
    unsafe {
        assert_eq!(c.get_unchecked(), &92);
    }
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn sync_once_cell_drop() {
    static DROP_CNT: AtomicUsize = AtomicUsize::new(0);
    struct Dropper;
    impl Drop for Dropper {
        fn drop(&mut self) {
            DROP_CNT.fetch_add(1, SeqCst);
        }
    }

    let x = OnceLock::new();
    spawn_and_wait(move || {
        x.get_or_init(|| Dropper);
        assert_eq!(DROP_CNT.load(SeqCst), 0);
        drop(x);
    });

    assert_eq!(DROP_CNT.load(SeqCst), 1);
}

#[test]
fn sync_once_cell_drop_empty() {
    let x = OnceLock::<String>::new();
    drop(x);
}

#[test]
fn clone() {
    let s = OnceLock::new();
    let c = s.clone();
    assert!(c.get().is_none());

    s.set("hello".to_string()).unwrap();
    let c = s.clone();
    assert_eq!(c.get().map(String::as_str), Some("hello"));
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn get_or_try_init() {
    let cell: OnceLock<String> = OnceLock::new();
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
    assert_eq!(OnceLock::from("value").get(), Some(&"value"));
    assert_ne!(OnceLock::from("foo").get(), Some(&"bar"));
}

#[test]
fn partialeq_impl() {
    assert!(OnceLock::from("value") == OnceLock::from("value"));
    assert!(OnceLock::from("foo") != OnceLock::from("bar"));

    assert!(OnceLock::<String>::new() == OnceLock::new());
    assert!(OnceLock::<String>::new() != OnceLock::from("value".to_owned()));
}

#[test]
fn into_inner() {
    let cell: OnceLock<String> = OnceLock::new();
    assert_eq!(cell.into_inner(), None);
    let cell = OnceLock::new();
    cell.set("hello".to_string()).unwrap();
    assert_eq!(cell.into_inner(), Some("hello".to_string()));
}

#[test]
fn is_sync_send() {
    fn assert_traits<T: Send + Sync>() {}
    assert_traits::<OnceLock<String>>();
}

#[test]
fn eval_once_macro() {
    macro_rules! eval_once {
        (|| -> $ty:ty {
            $($body:tt)*
        }) => {{
            static ONCE_CELL: OnceLock<$ty> = OnceLock::new();
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
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn sync_once_cell_does_not_leak_partially_constructed_boxes() {
    static ONCE_CELL: OnceLock<String> = OnceLock::new();

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
    let cell = OnceLock::new();
    {
        let s = String::new();
        cell.set(&s).unwrap();
    }
}
