use crate::cell::UnsafeCell;
use crate::sync::atomic::{AtomicBool, Ordering, spin_loop_hint};
use crate::ops::{Deref, DerefMut};

/// Trivial spinlock-based implementation of `sync::Mutex`.
// FIXME: Perhaps use Intel TSX to avoid locking?
#[derive(Default)]
pub struct SpinMutex<T> {
    value: UnsafeCell<T>,
    lock: AtomicBool,
}

unsafe impl<T: Send> Send for SpinMutex<T> {}
unsafe impl<T: Send> Sync for SpinMutex<T> {}

pub struct SpinMutexGuard<'a, T: 'a> {
    mutex: &'a SpinMutex<T>,
}

impl<'a, T> !Send for SpinMutexGuard<'a, T> {}
unsafe impl<'a, T: Sync> Sync for SpinMutexGuard<'a, T> {}

impl<T> SpinMutex<T> {
    pub const fn new(value: T) -> Self {
        SpinMutex {
            value: UnsafeCell::new(value),
            lock: AtomicBool::new(false)
        }
    }

    #[inline(always)]
    pub fn lock(&self) -> SpinMutexGuard<'_, T> {
        loop {
            match self.try_lock() {
                None => while self.lock.load(Ordering::Relaxed) {
                    spin_loop_hint()
                },
                Some(guard) => return guard
            }
        }
    }

    #[inline(always)]
    pub fn try_lock(&self) -> Option<SpinMutexGuard<'_, T>> {
        if !self.lock.compare_and_swap(false, true, Ordering::Acquire) {
            Some(SpinMutexGuard {
                mutex: self,
            })
        } else {
            None
        }
    }
}

/// Lock the Mutex or return false.
pub macro try_lock_or_false {
    ($e:expr) => {
        if let Some(v) = $e.try_lock() {
            v
        } else {
            return false
        }
    }
}

impl<'a, T> Deref for SpinMutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            &*self.mutex.value.get()
        }
    }
}

impl<'a, T> DerefMut for SpinMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe {
            &mut*self.mutex.value.get()
        }
    }
}

impl<'a, T> Drop for SpinMutexGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.lock.store(false, Ordering::Release)
    }
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)]

    use super::*;
    use crate::sync::Arc;
    use crate::thread;
    use crate::time::{SystemTime, Duration};

    #[test]
    fn sleep() {
        let mutex = Arc::new(SpinMutex::<i32>::default());
        let mutex2 = mutex.clone();
        let guard = mutex.lock();
        let t1 = thread::spawn(move || {
            *mutex2.lock() = 1;
        });

        // "sleep" for 50ms
        // FIXME: https://github.com/fortanix/rust-sgx/issues/31
        let start = SystemTime::now();
        let max = Duration::from_millis(50);
        while start.elapsed().unwrap() < max {}

        assert_eq!(*guard, 0);
        drop(guard);
        t1.join().unwrap();
        assert_eq!(*mutex.lock(), 1);
    }
}
