use std::fmt::Debug;
use std::ops::FnMut;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Condvar, MappedMutexGuard, Mutex, MutexGuard, TryLockError};
use std::{hint, mem, thread};

// Generates two test cases for both the poison and nonpoison locks.
//
// To write a test that uses both `poison::Mutex` and `nonpoison::Mutex`, simply call the macro
// `maybe_unwrap!(...)` on the result of `mutex.lock()`. For the `poison::Mutex`, it will unwrap the
// `LockResult`, but for the `nonpoison::Mutex` it will do nothing.
//
// The `poison` test will have the same name, but with a suffix of `_unwrap_poisoned`.
//
// See some of the tests below for examples.
macro_rules! nonpoison_and_poison_unwrap_test {
    (
        name: $name:ident,
        test_body: {$($test_body:tt)*}
    ) => {
        // Creates the nonpoison test.
        #[test]
        fn $name() {
            #[allow(unused_imports)]
            use ::std::sync::nonpoison::{Mutex, MappedMutexGuard, MutexGuard};

            #[allow(unused_macros)]
            macro_rules! maybe_unwrap {
                ($e:expr) => { $e };
            }

            $($test_body)*
        }

        // Creates the poison test with the suffix `_unwrap_poisoned`.
        #[test]
        fn ${concat($name, _unwrap_poisoned)}() {
            #[allow(unused_imports)]
            use ::std::sync::{Mutex, MappedMutexGuard, MutexGuard, LockResult};

            #[allow(unused_macros)]
            macro_rules! maybe_unwrap {
                ($e:expr) => { LockResult::unwrap($e) };
            }

            $($test_body)*
        }
    }
}

nonpoison_and_poison_unwrap_test!(
    name: smoke,
    test_body: {
        let m = Mutex::new(());
        drop(maybe_unwrap!(m.lock()));
        drop(maybe_unwrap!(m.lock()));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: lots_and_lots,
    test_body: {
        const J: u32 = 1000;
        const K: u32 = 3;

        let m = Arc::new(Mutex::new(0));

        fn inc(m: &Mutex<u32>) {
            for _ in 0..J {
                *maybe_unwrap!(m.lock()) += 1;
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
        assert_eq!(*maybe_unwrap!(m.lock()), J * K * 2);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: try_lock,
    test_body: {
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
        let m = Mutex::new(NonCopy(10));
        assert_eq!(maybe_unwrap!(m.into_inner()), NonCopy(10));
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_into_inner_drop,
    test_body: {
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
            let _inner = maybe_unwrap!(m.into_inner());
            assert_eq!(num_drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(num_drops.load(Ordering::SeqCst), 1);
    }
);

// TODO(connor): make this a double test after implementing `lock_value_accessors` on `nonpoison`.
#[test]
fn test_get_cloned() {
    #[derive(Clone, Eq, PartialEq, Debug)]
    struct Cloneable(i32);

    let m = Mutex::new(Cloneable(10));

    assert_eq!(m.get_cloned().unwrap(), Cloneable(10));
}

nonpoison_and_poison_unwrap_test!(
    name: test_get_mut,
    test_body: {
        let mut m = Mutex::new(NonCopy(10));
        *maybe_unwrap!(m.get_mut()) = NonCopy(20);
        assert_eq!(maybe_unwrap!(m.into_inner()), NonCopy(20));
    }
);

// TODO(connor): make this a double test after implementing `lock_value_accessors` on `nonpoison`.
#[test]
fn test_set() {
    fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
    where
        T: Debug + Eq,
    {
        let m = Mutex::new(init());

        assert_eq!(*m.lock().unwrap(), init());
        m.set(value()).unwrap();
        assert_eq!(*m.lock().unwrap(), value());
    }

    inner(|| NonCopy(10), || NonCopy(20));
    inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
}

// TODO(connor): make this a double test after implementing `lock_value_accessors` on `nonpoison`.
#[test]
fn test_replace() {
    fn inner<T>(mut init: impl FnMut() -> T, mut value: impl FnMut() -> T)
    where
        T: Debug + Eq,
    {
        let m = Mutex::new(init());

        assert_eq!(*m.lock().unwrap(), init());
        assert_eq!(m.replace(value()).unwrap(), init());
        assert_eq!(*m.lock().unwrap(), value());
    }

    inner(|| NonCopy(10), || NonCopy(20));
    inner(|| NonCopyNeedsDrop(10), || NonCopyNeedsDrop(20));
}

#[test]
fn test_mutex_arc_condvar() {
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

nonpoison_and_poison_unwrap_test!(
    name: test_mutex_arc_nested,
    test_body: {
        // Tests nested mutexes and access
        // to underlying data.
        let arc = Arc::new(Mutex::new(1));
        let arc2 = Arc::new(Mutex::new(arc));
        let (tx, rx) = channel();
        let _t = thread::spawn(move || {
            let lock = maybe_unwrap!(arc2.lock());
            let lock2 = maybe_unwrap!(lock.lock());
            assert_eq!(*lock2, 1);
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_mutex_unsized,
    test_body: {
        let mutex: &Mutex<[i32]> = &Mutex::new([1, 2, 3]);
        {
            let b = &mut *maybe_unwrap!(mutex.lock());
            b[0] = 4;
            b[2] = 5;
        }
        let comp: &[i32] = &[4, 2, 5];
        assert_eq!(&*maybe_unwrap!(mutex.lock()), comp);
    }
);

nonpoison_and_poison_unwrap_test!(
    name: test_mapping_mapped_guard,
    test_body: {
        let arr = [0; 4];
        let lock = Mutex::new(arr);
        let guard = maybe_unwrap!(lock.lock());
        let guard = MutexGuard::map(guard, |arr| &mut arr[..2]);
        let mut guard = MappedMutexGuard::map(guard, |slice| &mut slice[1..]);
        assert_eq!(guard.len(), 1);
        guard[0] = 42;
        drop(guard);
        assert_eq!(*maybe_unwrap!(lock.lock()), [0, 42, 0, 0]);
    }
);

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
