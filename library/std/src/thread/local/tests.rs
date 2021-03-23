use crate::cell::{Cell, UnsafeCell};
use crate::sync::mpsc::{channel, Sender};
use crate::thread::{self, LocalKey};
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
    run(&FOO);
    thread_local!(static FOO2: Cell<i32> = const { Cell::new(1) });
    run(&FOO2);

    fn run(key: &'static LocalKey<Cell<i32>>) {
        key.with(|f| {
            assert_eq!(f.get(), 1);
            f.set(2);
        });
        let t = thread::spawn(move || {
            key.with(|f| {
                assert_eq!(f.get(), 1);
            });
        });
        t.join().unwrap();

        key.with(|f| {
            assert_eq!(f.get(), 2);
        });
    }
}

#[test]
fn states() {
    struct Foo(&'static LocalKey<Foo>);
    impl Drop for Foo {
        fn drop(&mut self) {
            assert!(self.0.try_with(|_| ()).is_err());
        }
    }

    thread_local!(static FOO: Foo = Foo(&FOO));
    run(&FOO);
    thread_local!(static FOO2: Foo = const { Foo(&FOO2) });
    run(&FOO2);

    fn run(foo: &'static LocalKey<Foo>) {
        thread::spawn(move || {
            assert!(foo.try_with(|_| ()).is_ok());
        })
        .join()
        .unwrap();
    }
}

#[test]
fn smoke_dtor() {
    thread_local!(static FOO: UnsafeCell<Option<Foo>> = UnsafeCell::new(None));
    run(&FOO);
    thread_local!(static FOO2: UnsafeCell<Option<Foo>> = const { UnsafeCell::new(None) });
    run(&FOO2);

    fn run(key: &'static LocalKey<UnsafeCell<Option<Foo>>>) {
        let (tx, rx) = channel();
        let t = thread::spawn(move || unsafe {
            let mut tx = Some(tx);
            key.with(|f| {
                *f.get() = Some(Foo(tx.take().unwrap()));
            });
        });
        rx.recv().unwrap();
        t.join().unwrap();
    }
}

#[test]
fn circular() {
    struct S1(&'static LocalKey<UnsafeCell<Option<S1>>>, &'static LocalKey<UnsafeCell<Option<S2>>>);
    struct S2(&'static LocalKey<UnsafeCell<Option<S1>>>, &'static LocalKey<UnsafeCell<Option<S2>>>);
    thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));
    thread_local!(static K2: UnsafeCell<Option<S2>> = UnsafeCell::new(None));
    thread_local!(static K3: UnsafeCell<Option<S1>> = const { UnsafeCell::new(None) });
    thread_local!(static K4: UnsafeCell<Option<S2>> = const { UnsafeCell::new(None) });
    static mut HITS: usize = 0;

    impl Drop for S1 {
        fn drop(&mut self) {
            unsafe {
                HITS += 1;
                if self.1.try_with(|_| ()).is_err() {
                    assert_eq!(HITS, 3);
                } else {
                    if HITS == 1 {
                        self.1.with(|s| *s.get() = Some(S2(self.0, self.1)));
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
                assert!(self.0.try_with(|_| ()).is_ok());
                assert_eq!(HITS, 2);
                self.0.with(|s| *s.get() = Some(S1(self.0, self.1)));
            }
        }
    }

    thread::spawn(move || {
        drop(S1(&K1, &K2));
    })
    .join()
    .unwrap();

    unsafe {
        HITS = 0;
    }

    thread::spawn(move || {
        drop(S1(&K3, &K4));
    })
    .join()
    .unwrap();
}

#[test]
fn self_referential() {
    struct S1(&'static LocalKey<UnsafeCell<Option<S1>>>);

    thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));
    thread_local!(static K2: UnsafeCell<Option<S1>> = const { UnsafeCell::new(None) });

    impl Drop for S1 {
        fn drop(&mut self) {
            assert!(self.0.try_with(|_| ()).is_err());
        }
    }

    thread::spawn(move || unsafe {
        K1.with(|s| *s.get() = Some(S1(&K1)));
    })
    .join()
    .unwrap();

    thread::spawn(move || unsafe {
        K2.with(|s| *s.get() = Some(S1(&K2)));
    })
    .join()
    .unwrap();
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

#[test]
fn dtors_in_dtors_in_dtors_const_init() {
    struct S1(Sender<()>);
    thread_local!(static K1: UnsafeCell<Option<S1>> = const { UnsafeCell::new(None) });
    thread_local!(static K2: UnsafeCell<Option<Foo>> = const { UnsafeCell::new(None) });

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
