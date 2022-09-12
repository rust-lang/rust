use crate::cell::RefCell;
use crate::sync::Arc;
use crate::sys_common::remutex::{ReentrantMutex, ReentrantMutexGuard};
use crate::thread;

#[test]
fn smoke() {
    let m = ReentrantMutex::new(());
    {
        let a = m.lock();
        {
            let b = m.lock();
            {
                let c = m.lock();
                assert_eq!(*c, ());
            }
            assert_eq!(*b, ());
        }
        assert_eq!(*a, ());
    }
}

#[test]
fn is_mutex() {
    let m = Arc::new(ReentrantMutex::new(RefCell::new(0)));
    let m2 = m.clone();
    let lock = m.lock();
    let child = thread::spawn(move || {
        let lock = m2.lock();
        assert_eq!(*lock.borrow(), 4950);
    });
    for i in 0..100 {
        let lock = m.lock();
        *lock.borrow_mut() += i;
    }
    drop(lock);
    child.join().unwrap();
}

#[test]
fn trylock_works() {
    let m = Arc::new(ReentrantMutex::new(()));
    let m2 = m.clone();
    let _lock = m.try_lock();
    let _lock2 = m.try_lock();
    thread::spawn(move || {
        let lock = m2.try_lock();
        assert!(lock.is_none());
    })
    .join()
    .unwrap();
    let _lock3 = m.try_lock();
}

pub struct Answer<'a>(pub ReentrantMutexGuard<'a, RefCell<u32>>);
impl Drop for Answer<'_> {
    fn drop(&mut self) {
        *self.0.borrow_mut() = 42;
    }
}
