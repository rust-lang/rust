use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;

use super::nonpoison_and_poison_unwrap_test;

nonpoison_and_poison_unwrap_test!(
    name: smoke,
    test_body: {
        use locks::Condvar;

        let c = Condvar::new();
        c.notify_one();
        c.notify_all();
    }
);

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn poison_notify_one() {
    use std::sync::poison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let m2 = m.clone();
    let c = Arc::new(Condvar::new());
    let c2 = c.clone();

    let g = m.lock().unwrap();
    let _t = thread::spawn(move || {
        let _g = m2.lock().unwrap();
        c2.notify_one();
    });

    let g = c.wait(g).unwrap();
    drop(g);
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn nonpoison_notify_one() {
    use std::sync::nonpoison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let m2 = m.clone();
    let c = Arc::new(Condvar::new());
    let c2 = c.clone();

    let mut g = m.lock();
    let _t = thread::spawn(move || {
        let _g = m2.lock();
        c2.notify_one();
    });

    c.wait(&mut g);
    drop(g);
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn poison_notify_all() {
    use std::sync::poison::{Condvar, Mutex};

    const N: usize = 10;

    let data = Arc::new((Mutex::new(0), Condvar::new()));
    let (tx, rx) = channel();
    for _ in 0..N {
        let data = data.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            let &(ref lock, ref cond) = &*data;
            let mut cnt = lock.lock().unwrap();
            *cnt += 1;
            if *cnt == N {
                tx.send(()).unwrap();
            }
            while *cnt != 0 {
                cnt = cond.wait(cnt).unwrap();
            }
            tx.send(()).unwrap();
        });
    }
    drop(tx);

    let &(ref lock, ref cond) = &*data;
    rx.recv().unwrap();
    let mut cnt = lock.lock().unwrap();
    *cnt = 0;
    cond.notify_all();
    drop(cnt);

    for _ in 0..N {
        rx.recv().unwrap();
    }
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn nonpoison_notify_all() {
    use std::sync::nonpoison::{Condvar, Mutex};

    const N: usize = 10;

    let data = Arc::new((Mutex::new(0), Condvar::new()));
    let (tx, rx) = channel();
    for _ in 0..N {
        let data = data.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            let &(ref lock, ref cond) = &*data;
            let mut cnt = lock.lock();
            *cnt += 1;
            if *cnt == N {
                tx.send(()).unwrap();
            }
            while *cnt != 0 {
                cond.wait(&mut cnt);
            }
            tx.send(()).unwrap();
        });
    }
    drop(tx);

    let &(ref lock, ref cond) = &*data;
    rx.recv().unwrap();
    let mut cnt = lock.lock();
    *cnt = 0;
    cond.notify_all();
    drop(cnt);

    for _ in 0..N {
        rx.recv().unwrap();
    }
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn poison_test_mutex_arc_condvar() {
    use std::sync::poison::{Condvar, Mutex};

    struct Packet<T>(Arc<(Mutex<T>, Condvar)>);

    let packet = Packet(Arc::new((Mutex::new(false), Condvar::new())));
    let packet2 = Packet(packet.0.clone());

    let (tx, rx) = channel();

    let _t = thread::spawn(move || {
        // Wait until our parent has taken the lock.
        rx.recv().unwrap();
        let &(ref lock, ref cvar) = &*packet2.0;

        // Set the data to `true` and wake up our parent.
        let mut guard = lock.lock().unwrap();
        *guard = true;
        cvar.notify_one();
    });

    let &(ref lock, ref cvar) = &*packet.0;
    let mut guard = lock.lock().unwrap();
    // Wake up our child.
    tx.send(()).unwrap();

    // Wait until our child has set the data to `true`.
    assert!(!*guard);
    while !*guard {
        guard = cvar.wait(guard).unwrap();
    }
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn nonpoison_test_mutex_arc_condvar() {
    use std::sync::nonpoison::{Condvar, Mutex};

    struct Packet<T>(Arc<(Mutex<T>, Condvar)>);

    let packet = Packet(Arc::new((Mutex::new(false), Condvar::new())));
    let packet2 = Packet(packet.0.clone());

    let (tx, rx) = channel();

    let _t = thread::spawn(move || {
        // Wait until our parent has taken the lock.
        rx.recv().unwrap();
        let &(ref lock, ref cvar) = &*packet2.0;

        // Set the data to `true` and wake up our parent.
        let mut guard = lock.lock();
        *guard = true;
        cvar.notify_one();
    });

    let &(ref lock, ref cvar) = &*packet.0;
    let mut guard = lock.lock();
    // Wake up our child.
    tx.send(()).unwrap();

    // Wait until our child has set the data to `true`.
    assert!(!*guard);
    while !*guard {
        cvar.wait(&mut guard);
    }
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn poison_wait_while() {
    use std::sync::poison::{Condvar, Mutex};

    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair2 = pair.clone();

    // Inside of our lock, spawn a new thread, and then wait for it to start.
    thread::spawn(move || {
        let &(ref lock, ref cvar) = &*pair2;
        let mut started = lock.lock().unwrap();
        *started = true;
        // We notify the condvar that the value has changed.
        cvar.notify_one();
    });

    // Wait for the thread to start up.
    let &(ref lock, ref cvar) = &*pair;
    let guard = cvar.wait_while(lock.lock().unwrap(), |started| !*started).unwrap();
    assert!(*guard);
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn nonpoison_wait_while() {
    use std::sync::nonpoison::{Condvar, Mutex};

    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair2 = pair.clone();

    // Inside of our lock, spawn a new thread, and then wait for it to start.
    thread::spawn(move || {
        let &(ref lock, ref cvar) = &*pair2;
        let mut started = lock.lock();
        *started = true;
        // We notify the condvar that the value has changed.
        cvar.notify_one();
    });

    // Wait for the thread to start up.
    let &(ref lock, ref cvar) = &*pair;
    let mut guard = lock.lock();
    cvar.wait_while(&mut guard, |started| !*started);
    assert!(*guard);
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn poison_wait_timeout_wait() {
    use std::sync::poison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    loop {
        let g = m.lock().unwrap();
        let (_g, no_timeout) = c.wait_timeout(g, Duration::from_millis(1)).unwrap();
        // spurious wakeups mean this isn't necessarily true
        // so execute test again, if not timeout
        if !no_timeout.timed_out() {
            continue;
        }

        break;
    }
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn nonpoison_wait_timeout_wait() {
    use std::sync::nonpoison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    loop {
        let mut g = m.lock();
        let no_timeout = c.wait_timeout(&mut g, Duration::from_millis(1));
        // spurious wakeups mean this isn't necessarily true
        // so execute test again, if not timeout
        if !no_timeout.timed_out() {
            continue;
        }

        break;
    }
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn poison_wait_timeout_while_wait() {
    use std::sync::poison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    let g = m.lock().unwrap();
    let (_g, wait) = c.wait_timeout_while(g, Duration::from_millis(1), |_| true).unwrap();
    // no spurious wakeups. ensure it timed-out
    assert!(wait.timed_out());
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn nonpoison_wait_timeout_while_wait() {
    use std::sync::nonpoison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    let mut g = m.lock();
    let wait = c.wait_timeout_while(&mut g, Duration::from_millis(1), |_| true);
    // no spurious wakeups. ensure it timed-out
    assert!(wait.timed_out());
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn poison_wait_timeout_while_instant_satisfy() {
    use std::sync::poison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    let g = m.lock().unwrap();
    let (_g, wait) = c.wait_timeout_while(g, Duration::from_millis(0), |_| false).unwrap();
    // ensure it didn't time-out even if we were not given any time.
    assert!(!wait.timed_out());
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn nonpoison_wait_timeout_while_instant_satisfy() {
    use std::sync::nonpoison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    let mut g = m.lock();
    let wait = c.wait_timeout_while(&mut g, Duration::from_millis(0), |_| false);
    // ensure it didn't time-out even if we were not given any time.
    assert!(!wait.timed_out());
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn poison_wait_timeout_while_wake() {
    use std::sync::poison::{Condvar, Mutex};

    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair_copy = pair.clone();

    let &(ref m, ref c) = &*pair;
    let g = m.lock().unwrap();
    let _t = thread::spawn(move || {
        let &(ref lock, ref cvar) = &*pair_copy;
        let mut started = lock.lock().unwrap();
        thread::sleep(Duration::from_millis(1));
        *started = true;
        cvar.notify_one();
    });

    let (g2, wait) = c
        .wait_timeout_while(g, Duration::from_millis(u64::MAX), |&mut notified| !notified)
        .unwrap();
    // ensure it didn't time-out even if we were not given any time.
    assert!(!wait.timed_out());
    assert!(*g2);
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn nonpoison_wait_timeout_while_wake() {
    use std::sync::nonpoison::{Condvar, Mutex};

    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let pair_copy = pair.clone();

    let &(ref m, ref c) = &*pair;
    let mut g = m.lock();
    let _t = thread::spawn(move || {
        let &(ref lock, ref cvar) = &*pair_copy;
        let mut started = lock.lock();
        thread::sleep(Duration::from_millis(1));
        *started = true;
        cvar.notify_one();
    });

    let wait =
        c.wait_timeout_while(&mut g, Duration::from_millis(u64::MAX), |&mut notified| !notified);
    // ensure it didn't time-out even if we were not given any time.
    assert!(!wait.timed_out());
    assert!(*g);
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn poison_wait_timeout_wake() {
    use std::sync::poison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    loop {
        let g = m.lock().unwrap();

        let c2 = c.clone();
        let m2 = m.clone();

        let notified = Arc::new(AtomicBool::new(false));
        let notified_copy = notified.clone();

        let t = thread::spawn(move || {
            let _g = m2.lock().unwrap();
            thread::sleep(Duration::from_millis(1));
            notified_copy.store(true, Ordering::Relaxed);
            c2.notify_one();
        });

        let (g, timeout_res) = c.wait_timeout(g, Duration::from_millis(u64::MAX)).unwrap();
        assert!(!timeout_res.timed_out());
        // spurious wakeups mean this isn't necessarily true
        // so execute test again, if not notified
        if !notified.load(Ordering::Relaxed) {
            t.join().unwrap();
            continue;
        }
        drop(g);

        t.join().unwrap();

        break;
    }
}

#[test]
#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
fn nonpoison_wait_timeout_wake() {
    use std::sync::nonpoison::{Condvar, Mutex};

    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    loop {
        let mut g = m.lock();

        let c2 = c.clone();
        let m2 = m.clone();

        let notified = Arc::new(AtomicBool::new(false));
        let notified_copy = notified.clone();

        let t = thread::spawn(move || {
            let _g = m2.lock();
            thread::sleep(Duration::from_millis(1));
            notified_copy.store(true, Ordering::Relaxed);
            c2.notify_one();
        });

        let timeout_res = c.wait_timeout(&mut g, Duration::from_millis(u64::MAX));
        assert!(!timeout_res.timed_out());
        // spurious wakeups mean this isn't necessarily true
        // so execute test again, if not notified
        if !notified.load(Ordering::Relaxed) {
            t.join().unwrap();
            continue;
        }
        drop(g);

        t.join().unwrap();

        break;
    }
}

// Some platforms internally cast the timeout duration into nanoseconds.
// If they fail to consider overflow during the conversion (I'm looking
// at you, macOS), `wait_timeout` will return immediately and indicate a
// timeout for durations that are slightly longer than u64::MAX nanoseconds.
// `std` should guard against this by clamping the timeout.
// See #37440 for context.
#[test]
fn poison_timeout_nanoseconds() {
    use std::sync::poison::{Condvar, Mutex};

    let sent = Mutex::new(false);
    let cond = Condvar::new();

    thread::scope(|s| {
        s.spawn(|| {
            // Sleep so that the other thread has a chance to encounter the
            // timeout.
            thread::sleep(Duration::from_secs(2));
            *sent.lock().unwrap() = true;
            cond.notify_all();
        });

        let mut guard = sent.lock().unwrap();
        // Loop until `sent` is set by the thread to guard against spurious
        // wakeups. If the `wait_timeout` happens just before the signal by
        // the other thread, such a spurious wakeup might prevent the
        // miscalculated timeout from occurring, but this is basically just
        // a smoke test anyway.
        loop {
            if *guard {
                break;
            }

            // If there is internal overflow, this call will return almost
            // immediately, before the other thread has reached the `notify_all`,
            // and indicate a timeout.
            let (g, res) = cond
                .wait_timeout(guard, Duration::from_secs(u64::MAX.div_ceil(1_000_000_000)))
                .unwrap();
            assert!(!res.timed_out());
            guard = g;
        }
    })
}

#[test]
fn nonpoison_timeout_nanoseconds() {
    use std::sync::nonpoison::{Condvar, Mutex};

    let sent = Mutex::new(false);
    let cond = Condvar::new();

    thread::scope(|s| {
        s.spawn(|| {
            // Sleep so that the other thread has a chance to encounter the
            // timeout.
            thread::sleep(Duration::from_secs(2));
            sent.set(true);
            cond.notify_all();
        });

        let mut guard = sent.lock();
        // Loop until `sent` is set by the thread to guard against spurious
        // wakeups. If the `wait_timeout` happens just before the signal by
        // the other thread, such a spurious wakeup might prevent the
        // miscalculated timeout from occurring, but this is basically just
        // a smoke test anyway.
        loop {
            if *guard {
                break;
            }

            // If there is internal overflow, this call will return almost
            // immediately, before the other thread has reached the `notify_all`,
            // and indicate a timeout.
            let res = cond
                .wait_timeout(&mut guard, Duration::from_secs(u64::MAX.div_ceil(1_000_000_000)));
            assert!(!res.timed_out());
        }
    })
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_arc_condvar_poison() {
    use std::sync::poison::{Condvar, Mutex};

    struct Packet<T>(Arc<(Mutex<T>, Condvar)>);

    let packet = Packet(Arc::new((Mutex::new(1), Condvar::new())));
    let packet2 = Packet(packet.0.clone());
    let (tx, rx) = channel();

    let _t = thread::spawn(move || -> () {
        rx.recv().unwrap();
        let &(ref lock, ref cvar) = &*packet2.0;
        let _g = lock.lock().unwrap();
        cvar.notify_one();
        // Parent should fail when it wakes up.
        panic!();
    });

    let &(ref lock, ref cvar) = &*packet.0;
    let mut lock = lock.lock().unwrap();
    tx.send(()).unwrap();
    while *lock == 1 {
        match cvar.wait(lock) {
            Ok(l) => {
                lock = l;
                assert_eq!(*lock, 1);
            }
            Err(..) => break,
        }
    }
}
