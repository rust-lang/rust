use std::fmt::Debug;
use std::ops::FnMut;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Condvar, MappedMutexGuard, Mutex, MutexGuard, TryLockError};
use std::{hint, mem, thread};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Nonpoison & Poison Tests
////////////////////////////////////////////////////////////////////////////////////////////////////
use super::nonpoison_and_poison_unwrap_test;

nonpoison_and_poison_unwrap_test!(
    name: smoke,
    test_body: {
        use locks::Mutex;

        let m = Mutex::new(());
        drop(maybe_unwrap(m.lock()));
        drop(maybe_unwrap(m.lock()));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: lots_and_lots,
    test_body: {
        use locks::Mutex;

        const J: u32 = 1000;
        const K: u32 = 3;

        let m = Arc::new(Mutex::new(0));

        fn inc(m: &Mutex<u32>) {
            for _ in 0..J {
                *maybe_unwrap(m.lock()) += 1;
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
        assert_eq!(*maybe_unwrap(m.lock()), J * K * 2);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: try_lock,
    test_body: {
        use locks::Mutex;

        let m = Mutex::new(());
        *m.try_lock().unwrap() = ();
    }
);

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

nonpoison_and_poison_unwrap_test!(
    name: test_into_inner,
    test_body: {
        use locks::Mutex;

        let m = Mutex::new(NonCopy(10));
        assert_eq!(maybe_unwrap(m.into_inner()), NonCopy(10));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_into_inner_drop,
    test_body: {
        use locks::Mutex;

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
            let _inner = maybe_unwrap(m.into_inner());
            assert_eq!(num_drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(num_drops.load(Ordering::SeqCst), 1);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_get_mut,
    test_body: {
        use locks::Mutex;

        let mut m = Mutex::new(NonCopy(10));
        *maybe_unwrap(m.get_mut()) = NonCopy(20);
        assert_eq!(maybe_unwrap(m.into_inner()), NonCopy(20));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_get_cloned,
    test_body: {
        use locks::Mutex;

        #[derive(Clone, Eq, PartialEq, Debug)]
        struct Cloneable(i32);

        let m = Mutex::new(Cloneable(10));

        assert_eq!(maybe_unwrap(m.get_cloned()), Cloneable(10));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_set,
    test_body: {
        use locks::Mutex;

        fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
        where
            T: Debug + Eq,
        {
            let m = Mutex::new(init());

            assert_eq!(*maybe_unwrap(m.lock()), init());
            maybe_unwrap(m.set(value()));
            assert_eq!(*maybe_unwrap(m.lock()), value());
        }

        inner(|| NonCopy(10), || NonCopy(20));
        inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
    }
);

// Ensure that old values that are replaced by `set` are correctly dropped.
nonpoison_and_poison_unwrap_test!(
    name: test_set_drop,
    test_body: {
        use locks::Mutex;

        struct Foo(Arc<AtomicUsize>);
        impl Drop for Foo {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let num_drops = Arc::new(AtomicUsize::new(0));
        let m = Mutex::new(Foo(num_drops.clone()));
        assert_eq!(num_drops.load(Ordering::SeqCst), 0);

        let different = Foo(Arc::new(AtomicUsize::new(42)));
        maybe_unwrap(m.set(different));
        assert_eq!(num_drops.load(Ordering::SeqCst), 1);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_replace,
    test_body: {
        use locks::Mutex;

        fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
        where
            T: Debug + Eq,
        {
            let m = Mutex::new(init());

            assert_eq!(*maybe_unwrap(m.lock()), init());
            assert_eq!(maybe_unwrap(m.replace(value())), init());
            assert_eq!(*maybe_unwrap(m.lock()), value());
        }

        inner(|| NonCopy(10), || NonCopy(20));
        inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_mutex_arc_nested,
    test_body: {
        use locks::Mutex;

        // Tests nested mutexes and access
        // to underlying data.
        let arc = Arc::new(Mutex::new(1));
        let arc2 = Arc::new(Mutex::new(arc));
        let (tx, rx) = channel();
        let _t = thread::spawn(move || {
            let lock = maybe_unwrap(arc2.lock());
            let lock2 = maybe_unwrap(lock.lock());
            assert_eq!(*lock2, 1);
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_mutex_unsized,
    test_body: {
        use locks::Mutex;

        let mutex: &Mutex<[i32]> = &Mutex::new([1, 2, 3]);
        {
            let b = &mut *maybe_unwrap(mutex.lock());
            b[0] = 4;
            b[2] = 5;
        }
        let comp: &[i32] = &[4, 2, 5];
        assert_eq!(&*maybe_unwrap(mutex.lock()), comp);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_mapping_mapped_guard,
    test_body: {
        use locks::{Mutex, MutexGuard, MappedMutexGuard};

        let arr = [0; 4];
        let lock = Mutex::new(arr);
        let guard = maybe_unwrap(lock.lock());
        let guard = MutexGuard::map(guard, |arr| &mut arr[..2]);
        let mut guard = MappedMutexGuard::map(guard, |slice| &mut slice[1..]);
        assert_eq!(guard.len(), 1);
        guard[0] = 42;
        drop(guard);
        assert_eq!(*maybe_unwrap(lock.lock()), [0, 42, 0, 0]);
    }
);

#[cfg(panic = "unwind")] // Requires unwinding support.
nonpoison_and_poison_unwrap_test!(
    name: test_panics,
    test_body: {
        use locks::Mutex;

        let mutex = Mutex::new(42);

        let catch_unwind_result1 = panic::catch_unwind(AssertUnwindSafe(|| {
            let _guard1 = maybe_unwrap(mutex.lock());

            panic!("test panic with mutex once");
        }));
        assert!(catch_unwind_result1.is_err());

        let catch_unwind_result2 = panic::catch_unwind(AssertUnwindSafe(|| {
            let _guard2 = maybe_unwrap(mutex.lock());

            panic!("test panic with mutex twice");
        }));
        assert!(catch_unwind_result2.is_err());

        let catch_unwind_result3 = panic::catch_unwind(AssertUnwindSafe(|| {
            let _guard3 = maybe_unwrap(mutex.lock());

            panic!("test panic with mutex thrice");
        }));
        assert!(catch_unwind_result3.is_err());
    }
);

#[cfg(panic = "unwind")] // Requires unwinding support.
nonpoison_and_poison_unwrap_test!(
    name: test_mutex_arc_access_in_unwind,
    test_body: {
        use locks::Mutex;

        let arc = Arc::new(Mutex::new(1));
        let arc2 = arc.clone();
        let _ = thread::spawn(move || -> () {
            struct Unwinder {
                i: Arc<Mutex<i32>>,
            }
            impl Drop for Unwinder {
                fn drop(&mut self) {
                    *maybe_unwrap(self.i.lock()) += 1;
                }
            }
            let _u = Unwinder { i: arc2 };
            panic!();
        })
        .join();
        let lock = maybe_unwrap(arc.lock());
        assert_eq!(*lock, 2);
    }
);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Poison Tests
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Creates a mutex that is immediately poisoned.
fn new_poisoned_mutex<T>(value: T) -> Mutex<T> {
    let mutex = Mutex::new(value);

    let catch_unwind_result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _guard = mutex.lock().unwrap();

        panic!("test panic to poison mutex");
    }));

    assert!(catch_unwind_result.is_err());
    assert!(mutex.is_poisoned());

    mutex
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_into_inner_poison() {
    let m = new_poisoned_mutex(NonCopy(10));

    match m.into_inner() {
        Err(e) => assert_eq!(e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("into_inner of poisoned Mutex is Ok: {x:?}"),
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_get_cloned_poison() {
    #[derive(Clone, Eq, PartialEq, Debug)]
    struct Cloneable(i32);

    let m = new_poisoned_mutex(Cloneable(10));

    match m.get_cloned() {
        Err(e) => assert_eq!(e.into_inner(), ()),
        Ok(x) => panic!("get of poisoned Mutex is Ok: {x:?}"),
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_get_mut_poison() {
    let mut m = new_poisoned_mutex(NonCopy(10));

    match m.get_mut() {
        Err(e) => assert_eq!(*e.into_inner(), NonCopy(10)),
        Ok(x) => panic!("get_mut of poisoned Mutex is Ok: {x:?}"),
    }
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_set_poison() {
    fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
    where
        T: Debug + Eq,
    {
        let m = new_poisoned_mutex(init());

        match m.set(value()) {
            Err(e) => {
                assert_eq!(e.into_inner(), value());
                assert_eq!(m.into_inner().unwrap_err().into_inner(), init());
            }
            Ok(x) => panic!("set of poisoned Mutex is Ok: {x:?}"),
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
        let m = new_poisoned_mutex(init());

        match m.replace(value()) {
            Err(e) => {
                assert_eq!(e.into_inner(), value());
                assert_eq!(m.into_inner().unwrap_err().into_inner(), init());
            }
            Ok(x) => panic!("replace of poisoned Mutex is Ok: {x:?}"),
        }
    }

    inner(|| NonCopy(10), || NonCopy(20));
    inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
}

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn test_arc_condvar_poison() {
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

#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
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
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn panic_while_mapping_unlocked_poison() {
    let lock = Mutex::new(());

    let _ = panic::catch_unwind(|| {
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

    let _ = panic::catch_unwind(|| {
        let guard = lock.lock().unwrap();
        let _guard = MutexGuard::filter_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_lock() {
        Ok(_) => panic!("panicking in a MutexGuard::filter_map closure should poison the Mutex"),
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a MutexGuard::filter_map closure should unlock the mutex")
        }
        Err(TryLockError::Poisoned(_)) => {}
    }

    let _ = panic::catch_unwind(|| {
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

    let _ = panic::catch_unwind(|| {
        let guard = lock.lock().unwrap();
        let guard = MutexGuard::map::<(), _>(guard, |val| val);
        let _guard = MappedMutexGuard::filter_map::<(), _>(guard, |_| panic!());
    });

    match lock.try_lock() {
        Ok(_) => {
            panic!("panicking in a MappedMutexGuard::filter_map closure should poison the Mutex")
        }
        Err(TryLockError::WouldBlock) => {
            panic!("panicking in a MappedMutexGuard::filter_map closure should unlock the mutex")
        }
        Err(TryLockError::Poisoned(_)) => {}
    }

    drop(lock);
}

#[test]
fn test_mutex_with_mut() {
    let mutex = std::sync::nonpoison::Mutex::new(2);

    let result = mutex.with_mut(|value| {
        *value += 3;

        *value + 5
    });

    assert_eq!(*mutex.lock(), 5);
    assert_eq!(result, 10);
}
