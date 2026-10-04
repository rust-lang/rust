// MutexGuard<Cell<i32>> must not be Sync, that would be unsound.
use std::sync::Mutex;
use std::cell::Cell;

fn test_sync<T: Sync>(_t: T) {}

fn main()
{
    let m = Mutex::new(Cell::new(0i32));
    let guard = m.lock().unwrap();
    test_sync(guard);
    //~^ ERROR `Cell<i32>` cannot be shared between threads safely [E0277]
}
