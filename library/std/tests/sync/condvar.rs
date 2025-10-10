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

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
nonpoison_and_poison_unwrap_test!(
    name: notify_one,
    test_body: {
        use locks::{Condvar, Mutex};

        let m = Arc::new(Mutex::new(()));
        let m2 = m.clone();
        let c = Arc::new(Condvar::new());
        let c2 = c.clone();

        let g = maybe_unwrap(m.lock());
        let _t = thread::spawn(move || {
            let _g = maybe_unwrap(m2.lock());
            c2.notify_one();
        });
        let g = maybe_unwrap(c.wait(g));
        drop(g);
    }
);

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
nonpoison_and_poison_unwrap_test!(
    name: notify_all,
    test_body: {
        use locks::{Condvar, Mutex};

        const N: usize = 10;

        let data = Arc::new((Mutex::new(0), Condvar::new()));
        let (tx, rx) = channel();
        for _ in 0..N {
            let data = data.clone();
            let tx = tx.clone();
            thread::spawn(move || {
                let &(ref lock, ref cond) = &*data;
                let mut cnt = maybe_unwrap(lock.lock());
                *cnt += 1;
                if *cnt == N {
                    tx.send(()).unwrap();
                }
                while *cnt != 0 {
                    cnt = maybe_unwrap(cond.wait(cnt));
                }
                tx.send(()).unwrap();
            });
        }
        drop(tx);

        let &(ref lock, ref cond) = &*data;
        rx.recv().unwrap();
        let mut cnt = maybe_unwrap(lock.lock());
        *cnt = 0;
        cond.notify_all();
        drop(cnt);

        for _ in 0..N {
            rx.recv().unwrap();
        }
    }
);

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
nonpoison_and_poison_unwrap_test!(
    name: test_mutex_arc_condvar,
    test_body: {
        use locks::{Condvar, Mutex};

        struct Packet<T>(Arc<(Mutex<T>, Condvar)>);

        let packet = Packet(Arc::new((Mutex::new(false), Condvar::new())));
        let packet2 = Packet(packet.0.clone());

        let (tx, rx) = channel();

        let _t = thread::spawn(move || {
            // Wait until our parent has taken the lock.
            rx.recv().unwrap();
            let &(ref lock, ref cvar) = &*packet2.0;

            // Set the data to `true` and wake up our parent.
            let mut guard = maybe_unwrap(lock.lock());
            *guard = true;
            cvar.notify_one();
        });

        let &(ref lock, ref cvar) = &*packet.0;
        let mut guard = maybe_unwrap(lock.lock());
        // Wake up our child.
        tx.send(()).unwrap();

        // Wait until our child has set the data to `true`.
        assert!(!*guard);
        while !*guard {
            guard = maybe_unwrap(cvar.wait(guard));
        }
    }
);

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
nonpoison_and_poison_unwrap_test!(
    name: wait_while,
    test_body: {
        use locks::{Condvar, Mutex};

        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let pair2 = pair.clone();

        // Inside of our lock, spawn a new thread, and then wait for it to start.
        thread::spawn(move || {
            let &(ref lock, ref cvar) = &*pair2;
            let mut started = maybe_unwrap(lock.lock());
            *started = true;
            // We notify the condvar that the value has changed.
            cvar.notify_one();
        });

        // Wait for the thread to start up.
        let &(ref lock, ref cvar) = &*pair;
        let guard = cvar.wait_while(maybe_unwrap(lock.lock()), |started| !*started);
        assert!(*maybe_unwrap(guard));
    }
);

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
nonpoison_and_poison_unwrap_test!(
    name: wait_timeout_wait,
    test_body: {
        use locks::{Condvar, Mutex};

        let m = Arc::new(Mutex::new(()));
        let c = Arc::new(Condvar::new());

        loop {
            let g = maybe_unwrap(m.lock());
            let (_g, no_timeout) = maybe_unwrap(c.wait_timeout(g, Duration::from_millis(1)));
            // spurious wakeups mean this isn't necessarily true
            // so execute test again, if not timeout
            if !no_timeout.timed_out() {
                continue;
            }

            break;
        }
    }
);

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
nonpoison_and_poison_unwrap_test!(
    name: wait_timeout_while_wait,
    test_body: {
        use locks::{Condvar, Mutex};

        let m = Arc::new(Mutex::new(()));
        let c = Arc::new(Condvar::new());

        let g = maybe_unwrap(m.lock());
        let (_g, wait) = maybe_unwrap(c.wait_timeout_while(g, Duration::from_millis(1), |_| true));
        // no spurious wakeups. ensure it timed-out
        assert!(wait.timed_out());
    }
);

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
nonpoison_and_poison_unwrap_test!(
    name: wait_timeout_while_instant_satisfy,
    test_body: {
        use locks::{Condvar, Mutex};

        let m = Arc::new(Mutex::new(()));
        let c = Arc::new(Condvar::new());

        let g = maybe_unwrap(m.lock());
        let (_g, wait) =
            maybe_unwrap(c.wait_timeout_while(g, Duration::from_millis(0), |_| false));
        // ensure it didn't time-out even if we were not given any time.
        assert!(!wait.timed_out());
    }
);

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
nonpoison_and_poison_unwrap_test!(
    name: wait_timeout_while_wake,
    test_body: {
        use locks::{Condvar, Mutex};

        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let pair_copy = pair.clone();

        let &(ref m, ref c) = &*pair;
        let g = maybe_unwrap(m.lock());
        let _t = thread::spawn(move || {
            let &(ref lock, ref cvar) = &*pair_copy;
            let mut started = maybe_unwrap(lock.lock());
            thread::sleep(Duration::from_millis(1));
            *started = true;
            cvar.notify_one();
        });
        let (g2, wait) = maybe_unwrap(c.wait_timeout_while(
            g,
            Duration::from_millis(u64::MAX),
            |&mut notified| !notified
        ));
        // ensure it didn't time-out even if we were not given any time.
        assert!(!wait.timed_out());
        assert!(*g2);
    }
);

#[cfg(not(any(target_os = "emscripten", target_os = "wasi")))] // No threads.
nonpoison_and_poison_unwrap_test!(
    name: wait_timeout_wake,
    test_body: {
        use locks::{Condvar, Mutex};

        let m = Arc::new(Mutex::new(()));
        let c = Arc::new(Condvar::new());

        loop {
            let g = maybe_unwrap(m.lock());

            let c2 = c.clone();
            let m2 = m.clone();

            let notified = Arc::new(AtomicBool::new(false));
            let notified_copy = notified.clone();

            let t = thread::spawn(move || {
                let _g = maybe_unwrap(m2.lock());
                thread::sleep(Duration::from_millis(1));
                notified_copy.store(true, Ordering::Relaxed);
                c2.notify_one();
            });
            let (g, timeout_res) =
                maybe_unwrap(c.wait_timeout(g, Duration::from_millis(u64::MAX)));
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
);
