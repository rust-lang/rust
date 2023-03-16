//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

use std::cell::RefCell;
use std::thread;

struct TestCell {
    value: RefCell<u8>,
}

impl Drop for TestCell {
    fn drop(&mut self) {
        for _ in 0..10 {
            thread::yield_now();
        }
        println!("Dropping: {} (should be before 'Continue main 1').", *self.value.borrow())
    }
}

thread_local! {
    static A: TestCell = TestCell { value: RefCell::new(0) };
    static A_CONST: TestCell = const { TestCell { value: RefCell::new(10) } };
}

/// Check that destructors of the library thread locals are executed immediately
/// after a thread terminates.
fn check_destructors() {
    // We use the same value for both of them, since destructor order differs between Miri on Linux
    // (which uses `register_dtor_fallback`, in the end using a single pthread_key to manage a
    // thread-local linked list of dtors to call), real Linux rustc (which uses
    // `__cxa_thread_atexit_impl`), and Miri on Windows.
    thread::spawn(|| {
        A.with(|f| {
            assert_eq!(*f.value.borrow(), 0);
            *f.value.borrow_mut() = 8;
        });
        A_CONST.with(|f| {
            assert_eq!(*f.value.borrow(), 10);
            *f.value.borrow_mut() = 8;
        });
    })
    .join()
    .unwrap();
    println!("Continue main 1.")
}

struct JoinCell {
    value: RefCell<Option<thread::JoinHandle<u8>>>,
}

impl Drop for JoinCell {
    fn drop(&mut self) {
        for _ in 0..10 {
            thread::yield_now();
        }
        let join_handle = self.value.borrow_mut().take().unwrap();
        println!("Joining: {} (should be before 'Continue main 2').", join_handle.join().unwrap());
    }
}

thread_local! {
    static B: JoinCell = JoinCell { value: RefCell::new(None) };
}

/// Check that the destructor can be blocked joining another thread.
fn check_blocking() {
    thread::spawn(|| {
        B.with(|f| {
            assert!(f.value.borrow().is_none());
            let handle = thread::spawn(|| 7);
            *f.value.borrow_mut() = Some(handle);
        });
    })
    .join()
    .unwrap();
    println!("Continue main 2.");
    // Preempt the main thread so that the destructor gets executed and can join
    // the thread.
    thread::yield_now();
    thread::yield_now();
}

// This test tests that TLS destructors have run before the thread joins. The
// test has no false positives (meaning: if the test fails, there's actually
// an ordering problem). It may have false negatives, where the test passes but
// join is not guaranteed to be after the TLS destructors. However, false
// negatives should be exceedingly rare due to judicious use of
// thread::yield_now and running the test several times.
fn join_orders_after_tls_destructors() {
    use std::sync::atomic::{AtomicU8, Ordering};

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
    const FRESH: u8 = 0;
    const THREAD2_LAUNCHED: u8 = 1;
    const THREAD1_WAITING: u8 = 2;
    const MAIN_THREAD_RENDEZVOUS: u8 = 3;
    const THREAD2_JOINED: u8 = 4;
    static SYNC_STATE: AtomicU8 = AtomicU8::new(FRESH);

    for _ in 0..10 {
        SYNC_STATE.store(FRESH, Ordering::SeqCst);

        let jh = thread::Builder::new()
            .name("thread1".into())
            .spawn(move || {
                struct TlDrop;

                impl Drop for TlDrop {
                    fn drop(&mut self) {
                        let mut sync_state = SYNC_STATE.swap(THREAD1_WAITING, Ordering::SeqCst);
                        loop {
                            match sync_state {
                                THREAD2_LAUNCHED | THREAD1_WAITING => thread::yield_now(),
                                MAIN_THREAD_RENDEZVOUS => break,
                                THREAD2_JOINED =>
                                    panic!(
                                        "Thread 1 still running after thread 2 joined on thread 1"
                                    ),
                                v => unreachable!("sync state: {}", v),
                            }
                            sync_state = SYNC_STATE.load(Ordering::SeqCst);
                        }
                    }
                }

                thread_local! {
                    static TL_DROP: TlDrop = TlDrop;
                }

                TL_DROP.with(|_| {});

                loop {
                    match SYNC_STATE.load(Ordering::SeqCst) {
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
                assert_eq!(SYNC_STATE.swap(THREAD2_LAUNCHED, Ordering::SeqCst), FRESH);
                jh.join().unwrap();
                match SYNC_STATE.swap(THREAD2_JOINED, Ordering::SeqCst) {
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
                Ordering::SeqCst,
                Ordering::SeqCst,
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

fn dtors_in_dtors_in_dtors() {
    use std::cell::UnsafeCell;
    use std::sync::{Arc, Condvar, Mutex};

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
    // Note that this test will deadlock if TLS destructors aren't run (this
    // requires the destructor to be run to pass the test).
    signal.wait();
}

fn main() {
    check_destructors();
    check_blocking();
    join_orders_after_tls_destructors();
    dtors_in_dtors_in_dtors();
}
