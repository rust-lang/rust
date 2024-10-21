use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sync::mpsc::channel;
use crate::sync::{Arc, Condvar, Mutex};
use crate::thread;
use crate::time::Duration;

#[test]
fn smoke() {
    let c = Condvar::new();
    c.notify_one();
    c.notify_all();
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn notify_one() {
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
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn notify_all() {
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
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn wait_while() {
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
    let guard = cvar.wait_while(lock.lock().unwrap(), |started| !*started);
    assert!(*guard.unwrap());
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // condvar wait not supported
fn wait_timeout_wait() {
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
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // condvar wait not supported
fn wait_timeout_while_wait() {
    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    let g = m.lock().unwrap();
    let (_g, wait) = c.wait_timeout_while(g, Duration::from_millis(1), |_| true).unwrap();
    // no spurious wakeups. ensure it timed-out
    assert!(wait.timed_out());
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // condvar wait not supported
fn wait_timeout_while_instant_satisfy() {
    let m = Arc::new(Mutex::new(()));
    let c = Arc::new(Condvar::new());

    let g = m.lock().unwrap();
    let (_g, wait) = c.wait_timeout_while(g, Duration::from_millis(0), |_| false).unwrap();
    // ensure it didn't time-out even if we were not given any time.
    assert!(!wait.timed_out());
}

#[test]
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn wait_timeout_while_wake() {
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
#[cfg_attr(any(target_os = "emscripten", target_os = "wasi"), ignore)] // no threads
fn wait_timeout_wake() {
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
