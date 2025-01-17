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

#[derive(Clone, Eq, PartialEq, Debug)]
struct Cloneable(i32);

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
            let mut rng = crate::common::test_rng();
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
    let m = new_poisoned_rwlock(NonCopy(10));

    match m.into_inner() {
        Err(e) => assert_eq!(e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("into_inner of poisoned RwLock is Ok: {x:?}"),
    }
}

#[test]
fn test_get_cloned() {
    let m = RwLock::new(Cloneable(10));

    assert_eq!(m.get_cloned().unwrap(), Cloneable(10));
}

#[test]
fn test_get_cloned_poison() {
    let m = new_poisoned_rwlock(Cloneable(10));

    match m.get_cloned() {
        Err(e) => assert_eq!(e.into_inner(), ()),
        Ok(x) => panic!("get of poisoned RwLock is Ok: {x:?}"),
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
    let mut m = new_poisoned_rwlock(NonCopy(10));

    match m.get_mut() {
        Err(e) => assert_eq!(*e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("get_mut of poisoned RwLock is Ok: {x:?}"),
    }
}

#[test]
fn test_set() {
    fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
    where
        T: Debug + Eq,
    {
        let m = RwLock::new(init());

        assert_eq!(*m.read().unwrap(), init());
        m.set(value()).unwrap();
        assert_eq!(*m.read().unwrap(), value());
    }

    inner(|| NonCopy(10), || NonCopy(20));
    inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
}

#[test]
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
fn test_replace() {
    fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
    where
        T: Debug + Eq,
    {
        let m = RwLock::new(init());

        assert_eq!(*m.read().unwrap(), init());
        assert_eq!(m.replace(value()).unwrap(), init());
        assert_eq!(*m.read().unwrap(), value());
    }

    inner(|| NonCopy(10), || NonCopy(20));
    inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
}

#[test]
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
// FIXME: On macOS we use a provenance-incorrect implementation and Miri catches that issue.
// See <https://github.com/rust-lang/rust/issues/121950> for details.
#[cfg_attr(all(miri, target_os = "macos"), ignore)]
fn test_downgrade_observe() {
    // Taken from the test `test_rwlock_downgrade` from:
    // https://github.com/Amanieu/parking_lot/blob/master/src/rwlock.rs

    const W: usize = 20;
    const N: usize = if cfg!(miri) { 40 } else { 100 };

    // This test spawns `W` writer threads, where each will increment a counter `N` times, ensuring
    // that the value they wrote has not changed after downgrading.

    let rw = Arc::new(RwLock::new(0));

    // Spawn the writers that will do `W * N` operations and checks.
    let handles: Vec<_> = (0..W)
        .map(|_| {
            let rw = rw.clone();
            thread::spawn(move || {
                for _ in 0..N {
                    // Increment the counter.
                    let mut write_guard = rw.write().unwrap();
                    *write_guard += 1;
                    let cur_val = *write_guard;

                    // Downgrade the lock to read mode, where the value protected cannot be modified.
                    let read_guard = RwLockWriteGuard::downgrade(write_guard);
                    assert_eq!(cur_val, *read_guard);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(*rw.read().unwrap(), W * N);
}

#[test]
// FIXME: On macOS we use a provenance-incorrect implementation and Miri catches that issue.
// See <https://github.com/rust-lang/rust/issues/121950> for details.
#[cfg_attr(all(miri, target_os = "macos"), ignore)]
fn test_downgrade_atomic() {
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
    let mut main_write_guard = rwlock.write().unwrap();

    // Spawn all of the evil writer threads. They will each increment the protected value by 1.
    let handles: Vec<_> = (0..W)
        .map(|_| {
            let rwlock = rwlock.clone();
            thread::spawn(move || {
                // Will go to sleep since the main thread initially has the write lock.
                let mut evil_guard = rwlock.write().unwrap();
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

    // If the above is not atomic, then it would be possible for an evil thread to get in front of
    // this read and change the value to be non-negative.
    assert_eq!(*main_read_guard, NEW_VALUE, "`downgrade` was not atomic");

    // Drop the main read guard and allow the evil writer threads to start incrementing.
    drop(main_read_guard);

    for handle in handles {
        handle.join().unwrap();
    }

    let final_check = rwlock.read().unwrap();
    assert_eq!(*final_check, W as i32 + NEW_VALUE);
}
