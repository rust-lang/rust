use crate::boxed::Box;
use crate::cell::RefCell;
use crate::pin::Pin;
use crate::sync::Arc;
use crate::sys_common::remutex::{ReentrantMutex, ReentrantMutexGuard};
use crate::thread;

#[test]
fn smoke() {
    let m = unsafe {
        let mut m = Box::pin(ReentrantMutex::new(()));
        m.as_mut().init();
        m
    };
    let m = m.as_ref();
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
    let m = unsafe {
        // FIXME: Simplify this if Arc gets a Arc::get_pin_mut.
        let mut m = Arc::new(ReentrantMutex::new(RefCell::new(0)));
        Pin::new_unchecked(Arc::get_mut_unchecked(&mut m)).init();
        Pin::new_unchecked(m)
    };
    let m2 = m.clone();
    let lock = m.as_ref().lock();
    let child = thread::spawn(move || {
        let lock = m2.as_ref().lock();
        assert_eq!(*lock.borrow(), 4950);
    });
    for i in 0..100 {
        let lock = m.as_ref().lock();
        *lock.borrow_mut() += i;
    }
    drop(lock);
    child.join().unwrap();
}

#[test]
fn trylock_works() {
    let m = unsafe {
        // FIXME: Simplify this if Arc gets a Arc::get_pin_mut.
        let mut m = Arc::new(ReentrantMutex::new(()));
        Pin::new_unchecked(Arc::get_mut_unchecked(&mut m)).init();
        Pin::new_unchecked(m)
    };
    let m2 = m.clone();
    let _lock = m.as_ref().try_lock();
    let _lock2 = m.as_ref().try_lock();
    thread::spawn(move || {
        let lock = m2.as_ref().try_lock();
        assert!(lock.is_none());
    })
    .join()
    .unwrap();
    let _lock3 = m.as_ref().try_lock();
}

pub struct Answer<'a>(pub ReentrantMutexGuard<'a, RefCell<u32>>);
impl Drop for Answer<'_> {
    fn drop(&mut self) {
        *self.0.borrow_mut() = 42;
    }
}
