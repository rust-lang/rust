use crate::cell::{Cell, UnsafeCell};
use crate::sync::mpsc::{channel, Sender};
use crate::thread;
use crate::thread_local;

struct Foo(Sender<()>);

impl Drop for Foo {
    fn drop(&mut self) {
        let Foo(ref s) = *self;
        s.send(()).unwrap();
    }
}

#[test]
fn smoke_no_dtor() {
    thread_local!(static FOO: Cell<i32> = Cell::new(1));

    FOO.with(|f| {
        assert_eq!(f.get(), 1);
        f.set(2);
    });
    let (tx, rx) = channel();
    let _t = thread::spawn(move || {
        FOO.with(|f| {
            assert_eq!(f.get(), 1);
        });
        tx.send(()).unwrap();
    });
    rx.recv().unwrap();

    FOO.with(|f| {
        assert_eq!(f.get(), 2);
    });
}

#[test]
fn states() {
    struct Foo;
    impl Drop for Foo {
        fn drop(&mut self) {
            assert!(FOO.try_with(|_| ()).is_err());
        }
    }
    thread_local!(static FOO: Foo = Foo);

    thread::spawn(|| {
        assert!(FOO.try_with(|_| ()).is_ok());
    })
    .join()
    .ok()
    .expect("thread panicked");
}

#[test]
fn smoke_dtor() {
    thread_local!(static FOO: UnsafeCell<Option<Foo>> = UnsafeCell::new(None));

    let (tx, rx) = channel();
    let _t = thread::spawn(move || unsafe {
        let mut tx = Some(tx);
        FOO.with(|f| {
            *f.get() = Some(Foo(tx.take().unwrap()));
        });
    });
    rx.recv().unwrap();
}

#[test]
fn circular() {
    struct S1;
    struct S2;
    thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));
    thread_local!(static K2: UnsafeCell<Option<S2>> = UnsafeCell::new(None));
    static mut HITS: u32 = 0;

    impl Drop for S1 {
        fn drop(&mut self) {
            unsafe {
                HITS += 1;
                if K2.try_with(|_| ()).is_err() {
                    assert_eq!(HITS, 3);
                } else {
                    if HITS == 1 {
                        K2.with(|s| *s.get() = Some(S2));
                    } else {
                        assert_eq!(HITS, 3);
                    }
                }
            }
        }
    }
    impl Drop for S2 {
        fn drop(&mut self) {
            unsafe {
                HITS += 1;
                assert!(K1.try_with(|_| ()).is_ok());
                assert_eq!(HITS, 2);
                K1.with(|s| *s.get() = Some(S1));
            }
        }
    }

    thread::spawn(move || {
        drop(S1);
    })
    .join()
    .ok()
    .expect("thread panicked");
}

#[test]
fn self_referential() {
    struct S1;
    thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));

    impl Drop for S1 {
        fn drop(&mut self) {
            assert!(K1.try_with(|_| ()).is_err());
        }
    }

    thread::spawn(move || unsafe {
        K1.with(|s| *s.get() = Some(S1));
    })
    .join()
    .ok()
    .expect("thread panicked");
}

// Note that this test will deadlock if TLS destructors aren't run (this
// requires the destructor to be run to pass the test).
#[test]
fn dtors_in_dtors_in_dtors() {
    struct S1(Sender<()>);
    thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));
    thread_local!(static K2: UnsafeCell<Option<Foo>> = UnsafeCell::new(None));

    impl Drop for S1 {
        fn drop(&mut self) {
            let S1(ref tx) = *self;
            unsafe {
                let _ = K2.try_with(|s| *s.get() = Some(Foo(tx.clone())));
            }
        }
    }

    let (tx, rx) = channel();
    let _t = thread::spawn(move || unsafe {
        let mut tx = Some(tx);
        K1.with(|s| *s.get() = Some(S1(tx.take().unwrap())));
    });
    rx.recv().unwrap();
}
