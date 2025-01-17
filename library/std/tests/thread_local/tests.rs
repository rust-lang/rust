use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, Builder, LocalKey};
use std::thread_local;

#[derive(Clone, Default)]
struct Signal(Arc<(Mutex<bool>, Condvar)>);

impl Signal {
    fn notify(&self) {
        let (set, cvar) = &*self.0;
        *set.lock().unwrap() = true;
        cvar.notify_one();
    }

    fn wait(&self) {
        let (set, cvar) = &*self.0;
        let mut set = set.lock().unwrap();
        while !*set {
            set = cvar.wait(set).unwrap();
        }
    }
}

struct NotifyOnDrop(Signal);

impl Drop for NotifyOnDrop {
    fn drop(&mut self) {
        let NotifyOnDrop(ref f) = *self;
        f.notify();
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
    thread_local!(static FOO: UnsafeCell<Option<NotifyOnDrop>> = UnsafeCell::new(None));
    run(&FOO);
    thread_local!(static FOO2: UnsafeCell<Option<NotifyOnDrop>> = const { UnsafeCell::new(None) });
    run(&FOO2);

    fn run(key: &'static LocalKey<UnsafeCell<Option<NotifyOnDrop>>>) {
        let signal = Signal::default();
        let signal2 = signal.clone();
        let t = thread::spawn(move || unsafe {
            let mut signal = Some(signal2);
            key.with(|f| {
                *f.get() = Some(NotifyOnDrop(signal.take().unwrap()));
            });
        });
        signal.wait();
        t.join().unwrap();
    }
}

#[test]
fn circular() {
    // FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
    #![allow(static_mut_refs)]

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
    struct S1(Signal);
    thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));
    thread_local!(static K2: UnsafeCell<Option<NotifyOnDrop>> = UnsafeCell::new(None));

    impl Drop for S1 {
        fn drop(&mut self) {
            let S1(ref signal) = *self;
            unsafe {
                let _ = K2.try_with(|s| *s.get() = Some(NotifyOnDrop(signal.clone())));
            }
        }
    }

    let signal = Signal::default();
    let signal2 = signal.clone();
    let _t = thread::spawn(move || unsafe {
        let mut signal = Some(signal2);
        K1.with(|s| *s.get() = Some(S1(signal.take().unwrap())));
    });
    signal.wait();
}

#[test]
fn dtors_in_dtors_in_dtors_const_init() {
    struct S1(Signal);
    thread_local!(static K1: UnsafeCell<Option<S1>> = const { UnsafeCell::new(None) });
    thread_local!(static K2: UnsafeCell<Option<NotifyOnDrop>> = const { UnsafeCell::new(None) });

    impl Drop for S1 {
        fn drop(&mut self) {
            let S1(ref signal) = *self;
            unsafe {
                let _ = K2.try_with(|s| *s.get() = Some(NotifyOnDrop(signal.clone())));
            }
        }
    }

    let signal = Signal::default();
    let signal2 = signal.clone();
    let _t = thread::spawn(move || unsafe {
        let mut signal = Some(signal2);
        K1.with(|s| *s.get() = Some(S1(signal.take().unwrap())));
    });
    signal.wait();
}

// This test tests that TLS destructors have run before the thread joins. The
// test has no false positives (meaning: if the test fails, there's actually
// an ordering problem). It may have false negatives, where the test passes but
// join is not guaranteed to be after the TLS destructors. However, false
// negatives should be exceedingly rare due to judicious use of
// thread::yield_now and running the test several times.
#[test]
fn join_orders_after_tls_destructors() {
    // We emulate a synchronous MPSC rendezvous channel using only atomics and
    // thread::yield_now. We can't use std::mpsc as the implementation itself
    // may rely on thread locals.
    //
    // The basic state machine for an SPSC rendezvous channel is:
    //           FRESH -> THREAD1_WAITING -> MAIN_THREAD_RENDEZVOUS
    // where the first transition is done by the “receiving” thread and the 2nd
    // transition is done by the “sending” thread.
    //
    // We add an additional state `THREAD2_LAUNCHED` between `FRESH` and
    // `THREAD1_WAITING` to block until all threads are actually running.
    //
    // A thread that joins on the “receiving” thread completion should never
    // observe the channel in the `THREAD1_WAITING` state. If this does occur,
    // we switch to the “poison” state `THREAD2_JOINED` and panic all around.
    // (This is equivalent to “sending” from an alternate producer thread.)
    //
    // Relaxed memory ordering is fine because and spawn()/join() already provide all the
    // synchronization we need here.
    const FRESH: u8 = 0;
    const THREAD2_LAUNCHED: u8 = 1;
    const THREAD1_WAITING: u8 = 2;
    const MAIN_THREAD_RENDEZVOUS: u8 = 3;
    const THREAD2_JOINED: u8 = 4;
    static SYNC_STATE: AtomicU8 = AtomicU8::new(FRESH);

    for _ in 0..10 {
        SYNC_STATE.store(FRESH, Ordering::Relaxed);

        let jh = thread::Builder::new()
            .name("thread1".into())
            .spawn(move || {
                struct TlDrop;

                impl Drop for TlDrop {
                    fn drop(&mut self) {
                        let mut sync_state = SYNC_STATE.swap(THREAD1_WAITING, Ordering::Relaxed);
                        loop {
                            match sync_state {
                                THREAD2_LAUNCHED | THREAD1_WAITING => thread::yield_now(),
                                MAIN_THREAD_RENDEZVOUS => break,
                                THREAD2_JOINED => panic!(
                                    "Thread 1 still running after thread 2 joined on thread 1"
                                ),
                                v => unreachable!("sync state: {}", v),
                            }
                            sync_state = SYNC_STATE.load(Ordering::Relaxed);
                        }
                    }
                }

                thread_local! {
                    static TL_DROP: TlDrop = TlDrop;
                }

                TL_DROP.with(|_| {});

                loop {
                    match SYNC_STATE.load(Ordering::Relaxed) {
                        FRESH => thread::yield_now(),
                        THREAD2_LAUNCHED => break,
                        v => unreachable!("sync state: {}", v),
                    }
                }
            })
            .unwrap();

        let jh2 = thread::Builder::new()
            .name("thread2".into())
            .spawn(move || {
                assert_eq!(SYNC_STATE.swap(THREAD2_LAUNCHED, Ordering::Relaxed), FRESH);
                jh.join().unwrap();
                match SYNC_STATE.swap(THREAD2_JOINED, Ordering::Relaxed) {
                    MAIN_THREAD_RENDEZVOUS => return,
                    THREAD2_LAUNCHED | THREAD1_WAITING => {
                        panic!("Thread 2 running after thread 1 join before main thread rendezvous")
                    }
                    v => unreachable!("sync state: {:?}", v),
                }
            })
            .unwrap();

        loop {
            match SYNC_STATE.compare_exchange(
                THREAD1_WAITING,
                MAIN_THREAD_RENDEZVOUS,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(FRESH) => thread::yield_now(),
                Err(THREAD2_LAUNCHED) => thread::yield_now(),
                Err(THREAD2_JOINED) => {
                    panic!("Main thread rendezvous after thread 2 joined thread 1")
                }
                v => unreachable!("sync state: {:?}", v),
            }
        }
        jh2.join().unwrap();
    }
}

// Test that thread::current is still available in TLS destructors.
#[test]
fn thread_current_in_dtor() {
    // Go through one round of TLS destruction first.
    struct Defer;
    impl Drop for Defer {
        fn drop(&mut self) {
            RETRIEVE.with(|_| {});
        }
    }

    struct RetrieveName;
    impl Drop for RetrieveName {
        fn drop(&mut self) {
            *NAME.lock().unwrap() = Some(thread::current().name().unwrap().to_owned());
        }
    }

    static NAME: Mutex<Option<String>> = Mutex::new(None);

    thread_local! {
        static DEFER: Defer = const { Defer };
        static RETRIEVE: RetrieveName = const { RetrieveName };
    }

    Builder::new().name("test".to_owned()).spawn(|| DEFER.with(|_| {})).unwrap().join().unwrap();
    let name = NAME.lock().unwrap();
    let name = name.as_ref().unwrap();
    assert_eq!(name, "test");
}
