use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::mpsc::channel;
use crate::sync::{Arc, Condvar, MappedMutexGuard, Mutex, MutexGuard, TryLockError};
use crate::thread;

struct Packet<T>(Arc<(Mutex<T>, Condvar)>);

#[derive(Eq, PartialEq, Debug)]
struct NonCopy(i32);

#[test]
fn smoke() {
    let m = Mutex::new(());
    drop(m.lock().unwrap());
    drop(m.lock().unwrap());
}

#[test]
fn lots_and_lots() {
    const J: u32 = 1000;
    const K: u32 = 3;

    let m = Arc::new(Mutex::new(0));

    fn inc(m: &Mutex<u32>) {
        for _ in 0..J {
            *m.lock().unwrap() += 1;
        }
    }

    let (tx, rx) = channel();
    for _ in 0..K {
        let tx2 = tx.clone();
        let m2 = m.clone();
        thread::spawn(move || {
            inc(&m2);
            tx2.send(()).unwrap();
        });
        let tx2 = tx.clone();
        let m2 = m.clone();
        thread::spawn(move || {
            inc(&m2);
            tx2.send(()).unwrap();
        });
    }

    drop(tx);
    for _ in 0..2 * K {
        rx.recv().unwrap();
    }
    assert_eq!(*m.lock().unwrap(), J * K * 2);
}

#[test]
fn try_lock() {
    let m = Mutex::new(());
    *m.try_lock().unwrap() = ();
}

#[test]
fn test_into_inner() {
    let m = Mutex::new(NonCopy(10));
    assert_eq!(m.into_inner().unwrap(), NonCopy(10));
}

#[test]
fn test_into_inner_drop() {
    struct Foo(Arc<AtomicUsize>);
    impl Drop for Foo {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let num_drops = Arc::new(AtomicUsize::new(0));
    let m = Mutex::new(Foo(num_drops.clone()));
    assert_eq!(num_drops.load(Ordering::SeqCst), 0);
    {
        let _inner = m.into_inner().unwrap();
        assert_eq!(num_drops.load(Ordering::SeqCst), 0);
    }
    assert_eq!(num_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn test_into_inner_poison() {
    let m = Arc::new(Mutex::new(NonCopy(10)));
    let m2 = m.clone();
    let _ = thread::spawn(move || {
        let _lock = m2.lock().unwrap();
        panic!("test panic in inner thread to poison mutex");
    })
    .join();

    assert!(m.is_poisoned());
    match Arc::try_unwrap(m).unwrap().into_inner() {
        Err(e) => assert_eq!(e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("into_inner of poisoned Mutex is Ok: {x:?}"),
    }
}

#[test]
fn test_get_mut() {
    let mut m = Mutex::new(NonCopy(10));
    *m.get_mut().unwrap() = NonCopy(20);
    assert_eq!(m.into_inner().unwrap(), NonCopy(20));
}

#[test]
fn test_get_mut_poison() {
    let m = Arc::new(Mutex::new(NonCopy(10)));
    let m2 = m.clone();
    let _ = thread::spawn(move || {
        let _lock = m2.lock().unwrap();
        panic!("test panic in inner thread to poison mutex");
    })
    .join();

    assert!(m.is_poisoned());
    match Arc::try_unwrap(m).unwrap().get_mut() {
        Err(e) => assert_eq!(*e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("get_mut of poisoned Mutex is Ok: {x:?}"),
    }
}

#[test]
fn test_mutex_arc_condvar() {
    let packet = Packet(Arc::new((Mutex::new(false), Condvar::new())));
    let packet2 = Packet(packet.0.clone());
    let (tx, rx) = channel();
    let _t = thread::spawn(move || {
        // wait until parent gets in
        rx.recv().unwrap();
        let &(ref lock, ref cvar) = &*packet2.0;
        let mut lock = lock.lock().unwrap();
        *lock = true;
        cvar.notify_one();
    });

    let &(ref lock, ref cvar) = &*packet.0;
    let mut lock = lock.lock().unwrap();
    tx.send(()).unwrap();
    assert!(!*lock);
    while !*lock {
        lock = cvar.wait(lock).unwrap();
    }
}

#[test]
fn test_arc_condvar_poison() {
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

#[test]
fn test_mutex_arc_poison() {
    let arc = Arc::new(Mutex::new(1));
    assert!(!arc.is_poisoned());
    let arc2 = arc.clone();
    let _ = thread::spawn(move || {
        let lock = arc2.lock().unwrap();
        assert_eq!(*lock, 2); // deliberate assertion failure to poison the mutex
    })
    .join();
    assert!(arc.lock().is_err());
    assert!(arc.is_poisoned());
}

#[test]
fn test_mutex_arc_poison_mapped() {
    let arc = Arc::new(Mutex::new(1));
    assert!(!arc.is_poisoned());
    let arc2 = arc.clone();
    let _ = thread::spawn(move || {
        let lock = arc2.lock().unwrap();
        let lock = MutexGuard::map(lock, |val| val);
        assert_eq!(*lock, 2); // deliberate assertion failure to poison the mutex
    })
    .join();
    assert!(arc.lock().is_err());
    assert!(arc.is_poisoned());
}

#[test]
fn test_mutex_arc_nested() {
    // Tests nested mutexes and access
    // to underlying data.
    let arc = Arc::new(Mutex::new(1));
    let arc2 = Arc::new(Mutex::new(arc));
    let (tx, rx) = channel();
    let _t = thread::spawn(move || {
        let lock = arc2.lock().unwrap();
        let lock2 = lock.lock().unwrap();
        assert_eq!(*lock2, 1);
        tx.send(()).unwrap();
    });
    rx.recv().unwrap();
}

#[test]
fn test_mutex_arc_access_in_unwind() {
    let arc = Arc::new(Mutex::new(1));
    let arc2 = arc.clone();
    let _ = thread::spawn(move || -> () {
        struct Unwinder {
            i: Arc<Mutex<i32>>,
        }
        impl Drop for Unwinder {
            fn drop(&mut self) {
                *self.i.lock().unwrap() += 1;
            }
        }
        let _u = Unwinder { i: arc2 };
        panic!();
    })
    .join();
    let lock = arc.lock().unwrap();
    assert_eq!(*lock, 2);
}

#[test]
fn test_mutex_unsized() {
    let mutex: &Mutex<[i32]> = &Mutex::new([1, 2, 3]);
    {
        let b = &mut *mutex.lock().unwrap();
        b[0] = 4;
        b[2] = 5;
    }
    let comp: &[i32] = &[4, 2, 5];
    assert_eq!(&*mutex.lock().unwrap(), comp);
}

#[test]
fn test_mapping_mapped_guard() {
    let arr = [0; 4];
    let mut lock = Mutex::new(arr);
    let guard = lock.lock().unwrap();
    let guard = MutexGuard::map(guard, |arr| &mut arr[..2]);
    let mut guard = MappedMutexGuard::map(guard, |slice| &mut slice[1..]);
    assert_eq!(guard.len(), 1);
    guard[0] = 42;
    drop(guard);
    assert_eq!(*lock.get_mut().unwrap(), [0, 42, 0, 0]);
}

#[test]
fn panic_while_mapping_unlocked_poison() {
    let lock = Mutex::new(());

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.lock().unwrap();
        let _guard = MutexGuard::map::<(), _>(guard, |_| panic!());
    });

    match lock.try_lock() {
        Ok(_) => panic!("panicking in a MutexGuard::map closure should poison the Mutex"),
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a MutexGuard::map closure should unlock the mutex")
        }
        Err(TryLockError::Poisoned(_)) => {}
    }

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.lock().unwrap();
        let _guard = MutexGuard::try_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_lock() {
        Ok(_) => panic!("panicking in a MutexGuard::try_map closure should poison the Mutex"),
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a MutexGuard::try_map closure should unlock the mutex")
        }
        Err(TryLockError::Poisoned(_)) => {}
    }

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.lock().unwrap();
        let guard = MutexGuard::map::<(), _>(guard, |val| val);
        let _guard = MappedMutexGuard::map::<(), _>(guard, |_| panic!());
    });

    match lock.try_lock() {
        Ok(_) => panic!("panicking in a MappedMutexGuard::map closure should poison the Mutex"),
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a MappedMutexGuard::map closure should unlock the mutex")
        }
        Err(TryLockError::Poisoned(_)) => {}
    }

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.lock().unwrap();
        let guard = MutexGuard::map::<(), _>(guard, |val| val);
        let _guard = MappedMutexGuard::try_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_lock() {
        Ok(_) => panic!("panicking in a MappedMutexGuard::try_map closure should poison the Mutex"),
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a MappedMutexGuard::try_map closure should unlock the mutex")
        }
        Err(TryLockError::Poisoned(_)) => {}
    }

    drop(lock);
}
