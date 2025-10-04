use std::fmt::Debug;
use std::ops::FnMut;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::{
    Arc, MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
    TryLockError,
};
use std::{hint, mem, thread};

use rand::Rng;

#[derive(Eq, PartialEq, Debug)]
struct NonCopy(i32);

#[derive(Eq, PartialEq, Debug)]
struct NonCopyNeedsDrop(i32);

impl Drop for NonCopyNeedsDrop {
    fn drop(&mut self) {
        hint::black_box(());
    }
}

#[test]
fn test_needs_drop() {
    assert!(!mem::needs_drop::<NonCopy>());
    assert!(mem::needs_drop::<NonCopyNeedsDrop>());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Non-poison & Poison Tests
////////////////////////////////////////////////////////////////////////////////////////////////////
use super::nonpoison_and_poison_unwrap_test;

nonpoison_and_poison_unwrap_test!(
    name: smoke,
    test_body: {
        use locks::RwLock;

        let l = RwLock::new(());
        drop(maybe_unwrap(l.read()));
        drop(maybe_unwrap(l.write()));
        drop((maybe_unwrap(l.read()), maybe_unwrap(l.read())));
        drop(maybe_unwrap(l.write()));
    }
);

// FIXME: On macOS we use a provenance-incorrect implementation and Miri
// catches that issue with a chance of around 1/1000.
// See <https://github.com/rust-lang/rust/issues/121950> for details.
#[cfg(not(all(miri, target_os = "macos")))]
nonpoison_and_poison_unwrap_test!(
    name: frob,
    test_body: {
        use locks::RwLock;

        const N: u32 = 10;
        const M: usize = if cfg!(miri) { 100 } else { 1000 };

        let r = Arc::new(RwLock::new(()));

        let (tx, rx) = channel::<()>();
        for _ in 0..N {
            let tx = tx.clone();
            let r = r.clone();
            thread::spawn(move || {
                let mut rng = crate::common::test_rng();
                for _ in 0..M {
                    if rng.random_bool(1.0 / (N as f64)) {
                        drop(maybe_unwrap(r.write()));
                    } else {
                        drop(maybe_unwrap(r.read()));
                    }
                }
                drop(tx);
            });
        }
        drop(tx);
        let _ = rx.recv();
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_rw_arc,
    test_body: {
        use locks::RwLock;

        let arc = Arc::new(RwLock::new(0));
        let arc2 = arc.clone();
        let (tx, rx) = channel();

        thread::spawn(move || {
            let mut lock = maybe_unwrap(arc2.write());
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
                let lock = maybe_unwrap(arc3.read());
                assert!(*lock >= 0);
            }));
        }

        // Wait for children to pass their asserts
        for r in children {
            assert!(r.join().is_ok());
        }

        // Wait for writer to finish
        rx.recv().unwrap();
        let lock = maybe_unwrap(arc.read());
        assert_eq!(*lock, 10);
    }
);

#[cfg(panic = "unwind")] // Requires unwinding support.
nonpoison_and_poison_unwrap_test!(
    name: test_rw_arc_access_in_unwind,
    test_body: {
        use locks::RwLock;

        let arc = Arc::new(RwLock::new(1));
        let arc2 = arc.clone();
        let _ = thread::spawn(move || -> () {
            struct Unwinder {
                i: Arc<RwLock<isize>>,
            }
            impl Drop for Unwinder {
                fn drop(&mut self) {
                    let mut lock = maybe_unwrap(self.i.write());
                    *lock += 1;
                }
            }
            let _u = Unwinder { i: arc2 };
            panic!();
        })
        .join();
        let lock = maybe_unwrap(arc.read());
        assert_eq!(*lock, 2);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_rwlock_unsized,
    test_body: {
        use locks::RwLock;

        let rw: &RwLock<[i32]> = &RwLock::new([1, 2, 3]);
        {
            let b = &mut *maybe_unwrap(rw.write());
            b[0] = 4;
            b[2] = 5;
        }
        let comp: &[i32] = &[4, 2, 5];
        assert_eq!(&*maybe_unwrap(rw.read()), comp);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_into_inner,
    test_body: {
        use locks::RwLock;

        let m = RwLock::new(NonCopy(10));
        assert_eq!(maybe_unwrap(m.into_inner()), NonCopy(10));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_into_inner_drop,
    test_body: {
        use locks::RwLock;

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
            let _inner = maybe_unwrap(m.into_inner());
            assert_eq!(num_drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(num_drops.load(Ordering::SeqCst), 1);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_get_cloned,
    test_body: {
        use locks::RwLock;

        #[derive(Clone, Eq, PartialEq, Debug)]
        struct Cloneable(i32);

        let m = RwLock::new(Cloneable(10));

        assert_eq!(maybe_unwrap(m.get_cloned()), Cloneable(10));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_get_mut,
    test_body: {
        use locks::RwLock;

        let mut m = RwLock::new(NonCopy(10));
        *maybe_unwrap(m.get_mut()) = NonCopy(20);
        assert_eq!(maybe_unwrap(m.into_inner()), NonCopy(20));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_set,
    test_body: {
        use locks::RwLock;

        fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
        where
            T: Debug + Eq,
        {
            let m = RwLock::new(init());

            assert_eq!(*maybe_unwrap(m.read()), init());
            maybe_unwrap(m.set(value()));
            assert_eq!(*maybe_unwrap(m.read()), value());
        }

        inner(|| NonCopy(10), || NonCopy(20));
        inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_replace,
    test_body: {
        use locks::RwLock;

        fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
        where
            T: Debug + Eq,
        {
            let m = RwLock::new(init());

            assert_eq!(*maybe_unwrap(m.read()), init());
            assert_eq!(maybe_unwrap(m.replace(value())), init());
            assert_eq!(*maybe_unwrap(m.read()), value());
        }

        inner(|| NonCopy(10), || NonCopy(20));
        inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_read_guard_covariance,
    test_body: {
        use locks::{RwLock, RwLockReadGuard};

        fn do_stuff<'a>(_: RwLockReadGuard<'_, &'a i32>, _: &'a i32) {}
        let j: i32 = 5;
        let lock = RwLock::new(&j);
        {
            let i = 6;
            do_stuff(maybe_unwrap(lock.read()), &i);
        }
        drop(lock);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_mapped_read_guard_covariance,
    test_body: {
        use locks::{RwLock, RwLockReadGuard, MappedRwLockReadGuard};

        fn do_stuff<'a>(_: MappedRwLockReadGuard<'_, &'a i32>, _: &'a i32) {}
        let j: i32 = 5;
        let lock = RwLock::new((&j, &j));
        {
            let i = 6;
            let guard = maybe_unwrap(lock.read());
            let guard = RwLockReadGuard::map(guard, |(val, _val)| val);
            do_stuff(guard, &i);
        }
        drop(lock);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_downgrade_basic,
    test_body: {
        use locks::{RwLock, RwLockWriteGuard};

        let r = RwLock::new(());

        let write_guard = maybe_unwrap(r.write());
        let _read_guard = RwLockWriteGuard::downgrade(write_guard);
    }
);

// FIXME: On macOS we use a provenance-incorrect implementation and Miri catches that issue.
// See <https://github.com/rust-lang/rust/issues/121950> for details.
#[cfg(not(all(miri, target_os = "macos")))]
nonpoison_and_poison_unwrap_test!(
    name: test_downgrade_observe,
    test_body: {
        use locks::{RwLock, RwLockWriteGuard};

        // Inspired by the test `test_rwlock_downgrade` from:
        // https://github.com/Amanieu/parking_lot/blob/master/src/rwlock.rs

        const W: usize = 20;
        const N: usize = if cfg!(miri) { 40 } else { 100 };

        // This test spawns `W` writer threads, where each will increment a counter `N` times,
        // ensuring that the value they wrote has not changed after downgrading.

        let rw = Arc::new(RwLock::new(0));

        // Spawn the writers that will do `W * N` operations and checks.
        let handles: Vec<_> = (0..W)
            .map(|_| {
                let rw = rw.clone();
                thread::spawn(move || {
                    for _ in 0..N {
                        // Increment the counter.
                        let mut write_guard = maybe_unwrap(rw.write());
                        *write_guard += 1;
                        let cur_val = *write_guard;

                        // Downgrade the lock to read mode, where the value protected cannot be
                        // modified.
                        let read_guard = RwLockWriteGuard::downgrade(write_guard);
                        assert_eq!(cur_val, *read_guard);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(*maybe_unwrap(rw.read()), W * N);
    }
);

// FIXME: On macOS we use a provenance-incorrect implementation and Miri catches that issue.
// See <https://github.com/rust-lang/rust/issues/121950> for details.
#[cfg(not(all(miri, target_os = "macos")))]
nonpoison_and_poison_unwrap_test!(
    name: test_downgrade_atomic,
    test_body: {
        use locks::{RwLock, RwLockWriteGuard};

        const NEW_VALUE: i32 = -1;

        // This test checks that `downgrade` is atomic, meaning as soon as a write lock has been
        // downgraded, the lock must be in read mode and no other threads can take the write lock to
        // modify the protected value.

        // `W` is the number of evil writer threads.
        const W: usize = 20;
        let rwlock = Arc::new(RwLock::new(0));

        // Spawns many evil writer threads that will try and write to the locked value before the
        // initial writer (who has the exclusive lock) can read after it downgrades.
        // If the `RwLock` behaves correctly, then the initial writer should read the value it wrote
        // itself as no other thread should be able to mutate the protected value.

        // Put the lock in write mode, causing all future threads trying to access this go to sleep.
        let mut main_write_guard = maybe_unwrap(rwlock.write());

        // Spawn all of the evil writer threads. They will each increment the protected value by 1.
        let handles: Vec<_> = (0..W)
            .map(|_| {
                let rwlock = rwlock.clone();
                thread::spawn(move || {
                    // Will go to sleep since the main thread initially has the write lock.
                    let mut evil_guard = maybe_unwrap(rwlock.write());
                    *evil_guard += 1;
                })
            })
            .collect();

        // Wait for a good amount of time so that evil threads go to sleep.
        // Note: this is not strictly necessary...
        let eternity = std::time::Duration::from_millis(42);
        thread::sleep(eternity);

        // Once everyone is asleep, set the value to `NEW_VALUE`.
        *main_write_guard = NEW_VALUE;

        // Atomically downgrade the write guard into a read guard.
        let main_read_guard = RwLockWriteGuard::downgrade(main_write_guard);

        // If the above is not atomic, then it would be possible for an evil thread to get in front
        // of this read and change the value to be non-negative.
        assert_eq!(*main_read_guard, NEW_VALUE, "`downgrade` was not atomic");

        // Drop the main read guard and allow the evil writer threads to start incrementing.
        drop(main_read_guard);

        for handle in handles {
            handle.join().unwrap();
        }

        let final_check = maybe_unwrap(rwlock.read());
        assert_eq!(*final_check, W as i32 + NEW_VALUE);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_mapping_mapped_guard,
    test_body: {
        use locks::{
            RwLock, RwLockReadGuard, RwLockWriteGuard, MappedRwLockReadGuard, MappedRwLockWriteGuard
        };

        let arr = [0; 4];
        let mut lock = RwLock::new(arr);
        let guard = maybe_unwrap(lock.write());
        let guard = RwLockWriteGuard::map(guard, |arr| &mut arr[..2]);
        let mut guard = MappedRwLockWriteGuard::map(guard, |slice| &mut slice[1..]);
        assert_eq!(guard.len(), 1);
        guard[0] = 42;
        drop(guard);
        assert_eq!(*maybe_unwrap(lock.get_mut()), [0, 42, 0, 0]);

        let guard = maybe_unwrap(lock.read());
        let guard = RwLockReadGuard::map(guard, |arr| &arr[..2]);
        let guard = MappedRwLockReadGuard::map(guard, |slice| &slice[1..]);
        assert_eq!(*guard, [42]);
        drop(guard);
        assert_eq!(*maybe_unwrap(lock.get_mut()), [0, 42, 0, 0]);
    }
);

#[test]
fn nonpoison_test_rwlock_try_write() {
    use std::sync::nonpoison::{RwLock, RwLockReadGuard, WouldBlock};

    let lock = RwLock::new(0isize);
    let read_guard = lock.read();

    let write_result = lock.try_write();
    match write_result {
        Err(WouldBlock) => (),
        Ok(_) => assert!(false, "try_write should not succeed while read_guard is in scope"),
    }

    drop(read_guard);
    let mapped_read_guard = RwLockReadGuard::map(lock.read(), |_| &());

    let write_result = lock.try_write();
    match write_result {
        Err(WouldBlock) => (),
        Ok(_) => assert!(false, "try_write should not succeed while mapped_read_guard is in scope"),
    }

    drop(mapped_read_guard);
}

#[test]
fn poison_test_rwlock_try_write() {
    use std::sync::poison::{RwLock, RwLockReadGuard, TryLockError};

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

////////////////////////////////////////////////////////////////////////////////////////////////////
// Poison Tests
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Creates a rwlock that is immediately poisoned.
fn new_poisoned_rwlock<T>(value: T) -> RwLock<T> {
    let lock = RwLock::new(value);

    let catch_unwind_result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _guard = lock.write().unwrap();

        panic!("test panic to poison RwLock");
    }));

    assert!(catch_unwind_result.is_err());
    assert!(lock.is_poisoned());

    lock
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_into_inner_poison() {
    let m = new_poisoned_rwlock(NonCopy(10));

    match m.into_inner() {
        Err(e) => assert_eq!(e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("into_inner of poisoned RwLock is Ok: {x:?}"),
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_get_cloned_poison() {
    #[derive(Clone, Eq, PartialEq, Debug)]
    struct Cloneable(i32);

    let m = new_poisoned_rwlock(Cloneable(10));

    match m.get_cloned() {
        Err(e) => assert_eq!(e.into_inner(), ()),
        Ok(x) => panic!("get of poisoned RwLock is Ok: {x:?}"),
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_get_mut_poison() {
    let mut m = new_poisoned_rwlock(NonCopy(10));

    match m.get_mut() {
        Err(e) => assert_eq!(*e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("get_mut of poisoned RwLock is Ok: {x:?}"),
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_set_poison() {
    fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
    where
        T: Debug + Eq,
    {
        let m = new_poisoned_rwlock(init());

        match m.set(value()) {
            Err(e) => {
                assert_eq!(e.into_inner(), value());
                assert_eq!(m.into_inner().unwrap_err().into_inner(), init());
            }
            Ok(x) => panic!("set of poisoned RwLock is Ok: {x:?}"),
        }
    }

    inner(|| NonCopy(10), || NonCopy(20));
    inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_replace_poison() {
    fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
    where
        T: Debug + Eq,
    {
        let m = new_poisoned_rwlock(init());

        match m.replace(value()) {
            Err(e) => {
                assert_eq!(e.into_inner(), value());
                assert_eq!(m.into_inner().unwrap_err().into_inner(), init());
            }
            Ok(x) => panic!("replace of poisoned RwLock is Ok: {x:?}"),
        }
    }

    inner(|| NonCopy(10), || NonCopy(20));
    inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn panic_while_mapping_read_unlocked_no_poison() {
    let lock = RwLock::new(());

    let _ = panic::catch_unwind(|| {
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

    let _ = panic::catch_unwind(|| {
        let guard = lock.read().unwrap();
        let _guard = RwLockReadGuard::filter_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => {}
        Err(TryLockError::WouldBlock) => {
            panic!(
                "panicking in a RwLockReadGuard::filter_map closure should release the read lock"
            )
        }
        Err(TryLockError::Poisoned(_)) => {
            panic!(
                "panicking in a RwLockReadGuard::filter_map closure should not poison the RwLock"
            )
        }
    }

    let _ = panic::catch_unwind(|| {
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

    let _ = panic::catch_unwind(|| {
        let guard = lock.read().unwrap();
        let guard = RwLockReadGuard::map::<(), _>(guard, |val| val);
        let _guard = MappedRwLockReadGuard::filter_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => {}
        Err(TryLockError::WouldBlock) => panic!(
            "panicking in a MappedRwLockReadGuard::filter_map closure should release the read lock"
        ),
        Err(TryLockError::Poisoned(_)) => panic!(
            "panicking in a MappedRwLockReadGuard::filter_map closure should not poison the RwLock"
        ),
    }

    drop(lock);
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn panic_while_mapping_write_unlocked_poison() {
    let lock = RwLock::new(());

    let _ = panic::catch_unwind(|| {
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

    let _ = panic::catch_unwind(|| {
        let guard = lock.write().unwrap();
        let _guard = RwLockWriteGuard::filter_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => {
            panic!("panicking in a RwLockWriteGuard::filter_map closure should poison the RwLock")
        }
        Err(TryLockError::WouldBlock) => {
            panic!(
                "panicking in a RwLockWriteGuard::filter_map closure should release the write lock"
            )
        }
        Err(TryLockError::Poisoned(_)) => {}
    }

    let _ = panic::catch_unwind(|| {
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

    let _ = panic::catch_unwind(|| {
        let guard = lock.write().unwrap();
        let guard = RwLockWriteGuard::map::<(), _>(guard, |val| val);
        let _guard = MappedRwLockWriteGuard::filter_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_write() {
        Ok(_) => panic!(
            "panicking in a MappedRwLockWriteGuard::filter_map closure should poison the RwLock"
        ),
        Err(TryLockError::WouldBlock) => panic!(
            "panicking in a MappedRwLockWriteGuard::filter_map closure should release the write lock"
        ),
        Err(TryLockError::Poisoned(_)) => {}
    }

    drop(lock);
}

#[test]
fn test_rwlock_with() {
    let rwlock = std::sync::nonpoison::RwLock::new(2);
    let result = rwlock.with(|value| *value + 3);

    assert_eq!(result, 5);
}

#[test]
fn test_rwlock_with_mut() {
    let rwlock = std::sync::nonpoison::RwLock::new(2);

    let result = rwlock.with_mut(|value| {
        *value += 3;

        *value + 5
    });

    assert_eq!(*rwlock.read(), 5);
    assert_eq!(result, 10);
}
