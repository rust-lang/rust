use std::sync::Once;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::channel;
use std::time::Duration;
use std::{panic, thread};

#[test]
fn smoke_once() {
    static O: Once = Once::new();
    let mut a = 0;
    O.call_once(|| a += 1);
    assert_eq!(a, 1);
    O.call_once(|| a += 1);
    assert_eq!(a, 1);
}

#[test]
fn stampede_once() {
    static O: Once = Once::new();
    static mut RUN: bool = false;

    let (tx, rx) = channel();
    for _ in 0..10 {
        let tx = tx.clone();
        thread::spawn(move || {
            for _ in 0..4 {
                thread::yield_now()
            }
            unsafe {
                O.call_once(|| {
                    assert!(!RUN);
                    RUN = true;
                });
                assert!(RUN);
            }
            tx.send(()).unwrap();
        });
    }

    unsafe {
        O.call_once(|| {
            assert!(!RUN);
            RUN = true;
        });
        assert!(RUN);
    }

    for _ in 0..10 {
        rx.recv().unwrap();
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn poison_bad() {
    static O: Once = Once::new();

    // poison the once
    let t = panic::catch_unwind(|| {
        O.call_once(|| panic!());
    });
    assert!(t.is_err());

    // poisoning propagates
    let t = panic::catch_unwind(|| {
        O.call_once(|| {});
    });
    assert!(t.is_err());

    // we can subvert poisoning, however
    let mut called = false;
    O.call_once_force(|p| {
        called = true;
        assert!(p.is_poisoned())
    });
    assert!(called);

    // once any success happens, we stop propagating the poison
    O.call_once(|| {});
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn wait_for_force_to_finish() {
    static O: Once = Once::new();

    // poison the once
    let t = panic::catch_unwind(|| {
        O.call_once(|| panic!());
    });
    assert!(t.is_err());

    // make sure someone's waiting inside the once via a force
    let (tx1, rx1) = channel();
    let (tx2, rx2) = channel();
    let t1 = thread::spawn(move || {
        O.call_once_force(|p| {
            assert!(p.is_poisoned());
            tx1.send(()).unwrap();
            rx2.recv().unwrap();
        });
    });

    rx1.recv().unwrap();

    // put another waiter on the once
    let t2 = thread::spawn(|| {
        let mut called = false;
        O.call_once(|| {
            called = true;
        });
        assert!(!called);
    });

    tx2.send(()).unwrap();

    assert!(t1.join().is_ok());
    assert!(t2.join().is_ok());
}

#[test]
fn wait() {
    for _ in 0..50 {
        let val = AtomicBool::new(false);
        let once = Once::new();

        thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    once.wait();
                    assert!(val.load(Relaxed));
                });
            }

            once.call_once(|| val.store(true, Relaxed));
        });
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn wait_on_poisoned() {
    let once = Once::new();

    panic::catch_unwind(|| once.call_once(|| panic!())).unwrap_err();
    panic::catch_unwind(|| once.wait()).unwrap_err();
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn wait_force_on_poisoned() {
    let once = Once::new();

    thread::scope(|s| {
        panic::catch_unwind(|| once.call_once(|| panic!())).unwrap_err();

        s.spawn(|| {
            thread::sleep(Duration::from_millis(100));

            once.call_once_force(|_| {});
        });

        once.wait_force();
    })
}
