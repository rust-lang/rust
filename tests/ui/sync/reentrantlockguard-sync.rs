#![feature(reentrant_lock)]
use std::sync::ReentrantLock;
use std::cell::Cell;

// ReentrantLockGuard<Cell<i32>> must not be Sync, that would be unsound.

fn test_sync<T: Sync>(_t: T) {}

fn main()
{
    let m = ReentrantLock::new(Cell::new(0i32));
    let guard = m.lock();
    test_sync(guard);
    //~^ ERROR `Cell<i32>` cannot be shared between threads safely [E0277]
}
