use std::cell::RefCell;
use std::sync::{Arc, ReentrantLock};
use std::thread;

#[test]
fn smoke() {
    let l = ReentrantLock::new(());
    {
        let a = l.lock();
        {
            let b = l.lock();
            {
                let c = l.lock();
                assert_eq!(*c, ());
            }
            assert_eq!(*b, ());
        }
        assert_eq!(*a, ());
    }
}

#[test]
fn is_mutex() {
    let l = Arc::new(ReentrantLock::new(RefCell::new(0)));
    let l2 = l.clone();
    let lock = l.lock();
    let child = thread::spawn(move || {
        let lock = l2.lock();
        assert_eq!(*lock.borrow(), 4950);
    });
    for i in 0..100 {
        let lock = l.lock();
        *lock.borrow_mut() += i;
    }
    drop(lock);
    child.join().unwrap();
}

#[test]
fn trylock_works() {
    let l = Arc::new(ReentrantLock::new(()));
    let l2 = l.clone();
    let _lock = l.try_lock();
    let _lock2 = l.try_lock();
    thread::spawn(move || {
        let lock = l2.try_lock();
        assert!(lock.is_none());
    })
    .join()
    .unwrap();
    let _lock3 = l.try_lock();
}
