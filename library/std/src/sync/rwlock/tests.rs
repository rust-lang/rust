use rand::Rng;

use crate::hint::spin_loop;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::mpsc::channel;
use crate::sync::{
    Arc, Barrier, MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard,
    RwLockWriteGuard, TryLockError,
};
use crate::thread;

#[derive(Eq, PartialEq, Debug)]
struct NonCopy(i32);

#[test]
fn smoke() {
    let l = RwLock::new(());
    drop(l.read().unwrap());
    drop(l.write().unwrap());
    drop((l.read().unwrap(), l.read().unwrap()));
    drop(l.write().unwrap());
}

#[test]
// FIXME: On macOS we use a provenance-incorrect implementation and Miri
// catches that issue with a chance of around 1/1000.
// See <https://github.com/rust-lang/rust/issues/121950> for details.
#[cfg_attr(all(miri, target_os = "macos"), ignore)]
fn frob() {
    const N: u32 = 10;
    const M: usize = if cfg!(miri) { 100 } else { 1000 };

    let r = Arc::new(RwLock::new(()));

    let (tx, rx) = channel::<()>();
    for _ in 0..N {
        let tx = tx.clone();
        let r = r.clone();
        thread::spawn(move || {
            let mut rng = crate::test_helpers::test_rng();
            for _ in 0..M {
                if rng.gen_bool(1.0 / (N as f64)) {
                    drop(r.write().unwrap());
                } else {
                    drop(r.read().unwrap());
                }
            }
            drop(tx);
        });
    }
    drop(tx);
    let _ = rx.recv();
}

#[test]
fn test_rw_arc_poison_wr() {
    let arc = Arc::new(RwLock::new(1));
    let arc2 = arc.clone();
    let _: Result<(), _> = thread::spawn(move || {
        let _lock = arc2.write().unwrap();
        panic!();
    })
    .join();
    assert!(arc.read().is_err());
}

#[test]
fn test_rw_arc_poison_mapped_w_r() {
    let arc = Arc::new(RwLock::new(1));
    let arc2 = arc.clone();
    let _: Result<(), _> = thread::spawn(move || {
        let lock = arc2.write().unwrap();
        let _lock = RwLockWriteGuard::map(lock, |val| val);
        panic!();
    })
    .join();
    assert!(arc.read().is_err());
}

#[test]
fn test_rw_arc_poison_ww() {
    let arc = Arc::new(RwLock::new(1));
    assert!(!arc.is_poisoned());
    let arc2 = arc.clone();
    let _: Result<(), _> = thread::spawn(move || {
        let _lock = arc2.write().unwrap();
        panic!();
    })
    .join();
    assert!(arc.write().is_err());
    assert!(arc.is_poisoned());
}

#[test]
fn test_rw_arc_poison_mapped_w_w() {
    let arc = Arc::new(RwLock::new(1));
    let arc2 = arc.clone();
    let _: Result<(), _> = thread::spawn(move || {
        let lock = arc2.write().unwrap();
        let _lock = RwLockWriteGuard::map(lock, |val| val);
        panic!();
    })
    .join();
    assert!(arc.write().is_err());
    assert!(arc.is_poisoned());
}

#[test]
fn test_rw_arc_no_poison_rr() {
    let arc = Arc::new(RwLock::new(1));
    let arc2 = arc.clone();
    let _: Result<(), _> = thread::spawn(move || {
        let _lock = arc2.read().unwrap();
        panic!();
    })
    .join();
    let lock = arc.read().unwrap();
    assert_eq!(*lock, 1);
}

#[test]
fn test_rw_arc_no_poison_mapped_r_r() {
    let arc = Arc::new(RwLock::new(1));
    let arc2 = arc.clone();
    let _: Result<(), _> = thread::spawn(move || {
        let lock = arc2.read().unwrap();
        let _lock = RwLockReadGuard::map(lock, |val| val);
        panic!();
    })
    .join();
    let lock = arc.read().unwrap();
    assert_eq!(*lock, 1);
}

#[test]
fn test_rw_arc_no_poison_rw() {
    let arc = Arc::new(RwLock::new(1));
    let arc2 = arc.clone();
    let _: Result<(), _> = thread::spawn(move || {
        let _lock = arc2.read().unwrap();
        panic!()
    })
    .join();
    let lock = arc.write().unwrap();
    assert_eq!(*lock, 1);
}

#[test]
fn test_rw_arc_no_poison_mapped_r_w() {
    let arc = Arc::new(RwLock::new(1));
    let arc2 = arc.clone();
    let _: Result<(), _> = thread::spawn(move || {
        let lock = arc2.read().unwrap();
        let _lock = RwLockReadGuard::map(lock, |val| val);
        panic!();
    })
    .join();
    let lock = arc.write().unwrap();
    assert_eq!(*lock, 1);
}

#[test]
fn test_rw_arc() {
    let arc = Arc::new(RwLock::new(0));
    let arc2 = arc.clone();
    let (tx, rx) = channel();

    thread::spawn(move || {
        let mut lock = arc2.write().unwrap();
        for _ in 0..10 {
            let tmp = *lock;
            *lock = -1;
            thread::yield_now();
            *lock = tmp + 1;
        }
        tx.send(()).unwrap();
    });

    // Readers try to catch the writer in the act
    let mut children = Vec::new();
    for _ in 0..5 {
        let arc3 = arc.clone();
        children.push(thread::spawn(move || {
            let lock = arc3.read().unwrap();
            assert!(*lock >= 0);
        }));
    }

    // Wait for children to pass their asserts
    for r in children {
        assert!(r.join().is_ok());
    }

    // Wait for writer to finish
    rx.recv().unwrap();
    let lock = arc.read().unwrap();
    assert_eq!(*lock, 10);
}

#[test]
fn test_rw_arc_access_in_unwind() {
    let arc = Arc::new(RwLock::new(1));
    let arc2 = arc.clone();
    let _ = thread::spawn(move || -> () {
        struct Unwinder {
            i: Arc<RwLock<isize>>,
        }
        impl Drop for Unwinder {
            fn drop(&mut self) {
                let mut lock = self.i.write().unwrap();
                *lock += 1;
            }
        }
        let _u = Unwinder { i: arc2 };
        panic!();
    })
    .join();
    let lock = arc.read().unwrap();
    assert_eq!(*lock, 2);
}

#[test]
fn test_rwlock_unsized() {
    let rw: &RwLock<[i32]> = &RwLock::new([1, 2, 3]);
    {
        let b = &mut *rw.write().unwrap();
        b[0] = 4;
        b[2] = 5;
    }
    let comp: &[i32] = &[4, 2, 5];
    assert_eq!(&*rw.read().unwrap(), comp);
}

#[test]
fn test_rwlock_try_write() {
    let lock = RwLock::new(0isize);
    let read_guard = lock.read().unwrap();

    let write_result = lock.try_write();
    match write_result {
        Err(TryLockError::WouldBlock) => (),
        Ok(_) => assert!(false, "try_write should not succeed while read_guard is in scope"),
        Err(_) => assert!(false, "unexpected error"),
    }

    drop(read_guard);
    let mapped_read_guard = RwLockReadGuard::map(lock.read().unwrap(), |_| &());

    let write_result = lock.try_write();
    match write_result {
        Err(TryLockError::WouldBlock) => (),
        Ok(_) => assert!(false, "try_write should not succeed while mapped_read_guard is in scope"),
        Err(_) => assert!(false, "unexpected error"),
    }

    drop(mapped_read_guard);
}

#[test]
fn test_into_inner() {
    let m = RwLock::new(NonCopy(10));
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
    let m = RwLock::new(Foo(num_drops.clone()));
    assert_eq!(num_drops.load(Ordering::SeqCst), 0);
    {
        let _inner = m.into_inner().unwrap();
        assert_eq!(num_drops.load(Ordering::SeqCst), 0);
    }
    assert_eq!(num_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn test_into_inner_poison() {
    let m = Arc::new(RwLock::new(NonCopy(10)));
    let m2 = m.clone();
    let _ = thread::spawn(move || {
        let _lock = m2.write().unwrap();
        panic!("test panic in inner thread to poison RwLock");
    })
    .join();

    assert!(m.is_poisoned());
    match Arc::try_unwrap(m).unwrap().into_inner() {
        Err(e) => assert_eq!(e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("into_inner of poisoned RwLock is Ok: {x:?}"),
    }
}

#[test]
fn test_get_mut() {
    let mut m = RwLock::new(NonCopy(10));
    *m.get_mut().unwrap() = NonCopy(20);
    assert_eq!(m.into_inner().unwrap(), NonCopy(20));
}

#[test]
fn test_get_mut_poison() {
    let m = Arc::new(RwLock::new(NonCopy(10)));
    let m2 = m.clone();
    let _ = thread::spawn(move || {
        let _lock = m2.write().unwrap();
        panic!("test panic in inner thread to poison RwLock");
    })
    .join();

    assert!(m.is_poisoned());
    match Arc::try_unwrap(m).unwrap().get_mut() {
        Err(e) => assert_eq!(*e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("get_mut of poisoned RwLock is Ok: {x:?}"),
    }
}

#[test]
fn test_read_guard_covariance() {
    fn do_stuff<'a>(_: RwLockReadGuard<'_, &'a i32>, _: &'a i32) {}
    let j: i32 = 5;
    let lock = RwLock::new(&j);
    {
        let i = 6;
        do_stuff(lock.read().unwrap(), &i);
    }
    drop(lock);
}

#[test]
fn test_mapped_read_guard_covariance() {
    fn do_stuff<'a>(_: MappedRwLockReadGuard<'_, &'a i32>, _: &'a i32) {}
    let j: i32 = 5;
    let lock = RwLock::new((&j, &j));
    {
        let i = 6;
        let guard = lock.read().unwrap();
        let guard = RwLockReadGuard::map(guard, |(val, _val)| val);
        do_stuff(guard, &i);
    }
    drop(lock);
}

#[test]
fn test_mapping_mapped_guard() {
    let arr = [0; 4];
    let mut lock = RwLock::new(arr);
    let guard = lock.write().unwrap();
    let guard = RwLockWriteGuard::map(guard, |arr| &mut arr[..2]);
    let mut guard = MappedRwLockWriteGuard::map(guard, |slice| &mut slice[1..]);
    assert_eq!(guard.len(), 1);
    guard[0] = 42;
    drop(guard);
    assert_eq!(*lock.get_mut().unwrap(), [0, 42, 0, 0]);

    let guard = lock.read().unwrap();
    let guard = RwLockReadGuard::map(guard, |arr| &arr[..2]);
    let guard = MappedRwLockReadGuard::map(guard, |slice| &slice[1..]);
    assert_eq!(*guard, [42]);
    drop(guard);
    assert_eq!(*lock.get_mut().unwrap(), [0, 42, 0, 0]);
}

#[test]
fn panic_while_mapping_read_unlocked_no_poison() {
    let lock = RwLock::new(());

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.read().unwrap();
        let _guard = RwLockReadGuard::map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => {}
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a RwLockReadGuard::map closure should release the read lock")
        }
        Err(TryLockError::Poisoned(_)) => {
            panic!("panicking in a RwLockReadGuard::map closure should not poison the RwLock")
        }
    }

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.read().unwrap();
        let _guard = RwLockReadGuard::try_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => {}
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a RwLockReadGuard::try_map closure should release the read lock")
        }
        Err(TryLockError::Poisoned(_)) => {
            panic!("panicking in a RwLockReadGuard::try_map closure should not poison the RwLock")
        }
    }

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.read().unwrap();
        let guard = RwLockReadGuard::map::<(), _>(guard, |val| val);
        let _guard = MappedRwLockReadGuard::map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => {}
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a MappedRwLockReadGuard::map closure should release the read lock")
        }
        Err(TryLockError::Poisoned(_)) => {
            panic!("panicking in a MappedRwLockReadGuard::map closure should not poison the RwLock")
        }
    }

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.read().unwrap();
        let guard = RwLockReadGuard::map::<(), _>(guard, |val| val);
        let _guard = MappedRwLockReadGuard::try_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => {}
        Err(TryLockError::WouldBlock) => panic!(
            "panicking in a MappedRwLockReadGuard::try_map closure should release the read lock"
        ),
        Err(TryLockError::Poisoned(_)) => panic!(
            "panicking in a MappedRwLockReadGuard::try_map closure should not poison the RwLock"
        ),
    }

    drop(lock);
}

#[test]
fn panic_while_mapping_write_unlocked_poison() {
    let lock = RwLock::new(());

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.write().unwrap();
        let _guard = RwLockWriteGuard::map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => panic!("panicking in a RwLockWriteGuard::map closure should poison the RwLock"),
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a RwLockWriteGuard::map closure should release the write lock")
        }
        Err(TryLockError::Poisoned(_)) => {}
    }

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.write().unwrap();
        let _guard = RwLockWriteGuard::try_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => {
            panic!("panicking in a RwLockWriteGuard::try_map closure should poison the RwLock")
        }
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a RwLockWriteGuard::try_map closure should release the write lock")
        }
        Err(TryLockError::Poisoned(_)) => {}
    }

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.write().unwrap();
        let guard = RwLockWriteGuard::map::<(), _>(guard, |val| val);
        let _guard = MappedRwLockWriteGuard::map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => {
            panic!("panicking in a MappedRwLockWriteGuard::map closure should poison the RwLock")
        }
        Err(TryLockError::WouldBlock) => panic!(
            "panicking in a MappedRwLockWriteGuard::map closure should release the write lock"
        ),
        Err(TryLockError::Poisoned(_)) => {}
    }

    let _ = crate::panic::catch_unwind(|| {
        let guard = lock.write().unwrap();
        let guard = RwLockWriteGuard::map::<(), _>(guard, |val| val);
        let _guard = MappedRwLockWriteGuard::try_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => panic!(
            "panicking in a MappedRwLockWriteGuard::try_map closure should poison the RwLock"
        ),
        Err(TryLockError::WouldBlock) => panic!(
            "panicking in a MappedRwLockWriteGuard::try_map closure should release the write lock"
        ),
        Err(TryLockError::Poisoned(_)) => {}
    }

    drop(lock);
}

#[test]
fn test_downgrade_basic() {
    let r = RwLock::new(());

    let write_guard = r.write().unwrap();
    let _read_guard = RwLockWriteGuard::downgrade(write_guard);
}

#[test]
fn test_downgrade_frob() {
    const N: u32 = 10;
    const M: usize = if cfg!(miri) { 100 } else { 1000 };

    let r = Arc::new(RwLock::new(()));

    let (tx, rx) = channel::<()>();
    for _ in 0..N {
        let tx = tx.clone();
        let r = r.clone();
        thread::spawn(move || {
            let mut rng = crate::test_helpers::test_rng();
            for _ in 0..M {
                if rng.gen_bool(1.0 / (N as f64)) {
                    drop(RwLockWriteGuard::downgrade(r.write().unwrap()));
                } else {
                    drop(r.read().unwrap());
                }
            }
            drop(tx);
        });
    }
    drop(tx);
    let _ = rx.recv();
}

#[test]
fn test_downgrade_readers() {
    const R: usize = 16;
    const N: usize = 1000;

    // Starts up 1 writing thread and `R` reader threads.
    // The writer thread will constantly update the value inside the `RwLock`, and this test will
    // only pass if every reader observes all values between 0 and `N`.
    let r = Arc::new(RwLock::new(0));
    let b = Arc::new(Barrier::new(R + 1));

    // Create the writing thread.
    let r_writer = r.clone();
    let b_writer = b.clone();
    thread::spawn(move || {
        for i in 0..N {
            let mut write_guard = r_writer.write().unwrap();
            *write_guard = i;

            let read_guard = RwLockWriteGuard::downgrade(write_guard);
            assert_eq!(*read_guard, i);

            // Wait for all readers to observe the new value.
            b_writer.wait();
        }
    });

    for _ in 0..R {
        let r = r.clone();
        let b = b.clone();
        thread::spawn(move || {
            // Every reader thread needs to observe every value up to `N`.
            for i in 0..N {
                let read_guard = r.read().unwrap();
                assert_eq!(*read_guard, i);
                drop(read_guard);

                // Wait for everyone to read and for the writer to change the value again.
                b.wait();
                // Spin until the writer has changed the value.

                loop {
                    let read_guard = r.read().unwrap();
                    assert!(*read_guard >= i);

                    if *read_guard > i {
                        break;
                    }
                }
            }
        });
    }
}

#[test]
fn test_downgrade_atomic() {
    const R: usize = 16;

    let r = Arc::new(RwLock::new(0));
    // The number of reader threads that observe the correct value.
    let observers = Arc::new(AtomicUsize::new(0));

    let w = r.clone();
    let mut main_write_guard = w.write().unwrap();

    // While the current thread is holding the write lock, spawn several reader threads and an evil
    // writer thread.
    // Each of the threads will attempt to read the `RwLock` and go to sleep because we have the
    // write lock.
    // We need at least 1 reader thread to observe what the main thread writes, otherwise that means
    // the evil writer thread got in front of every single reader.

    // FIXME
    // Should we actually require that every reader observe the first change?
    // This is a matter of protocol rather than correctness...

    let mut reader_handles = Vec::with_capacity(R);

    for _ in 0..R {
        let r = r.clone();
        let observers = observers.clone();
        let handle = thread::spawn(move || {
            // Will go to sleep since the main thread initially has the write lock.
            let read_guard = r.read().unwrap();
            if *read_guard == 1 {
                observers.fetch_add(1, Ordering::Relaxed);
            }
        });

        reader_handles.push(handle);
    }

    let evil = r.clone();
    let evil_handle = thread::spawn(move || {
        // Will go to sleep since the main thread initially has the write lock.
        let mut evil_guard = evil.write().unwrap();
        *evil_guard = 2;
    });

    // FIXME Come up with a better way to make sure everyone is sleeping.
    // Make sure that everyone else is actually sleeping.
    let spin = 1000000;
    for _ in 0..spin {
        spin_loop();
    }

    // Once everyone is asleep, set the value to 1.
    *main_write_guard = 1;

    // Atomically downgrade the write guard into a read guard.
    // This should wake up all of the reader threads, and allow them to also take the read lock.
    let main_read_guard = RwLockWriteGuard::downgrade(main_write_guard);

    // If the above is not atomic, then it is possible for the evil thread to get in front of the
    // readers and change the value to 2 instead.
    assert_eq!(*main_read_guard, 1, "`downgrade` was not atomic");

    // By dropping all of the read guards, we allow the evil thread to make the change.
    drop(main_read_guard);

    for handle in reader_handles {
        handle.join().unwrap();
    }

    // Wait for the evil thread to set the value to 2.
    evil_handle.join().unwrap();

    let final_check = r.read().unwrap();
    assert_eq!(*final_check, 2);

    assert!(observers.load(Ordering::Relaxed) > 0, "No readers observed the correct value");
}
