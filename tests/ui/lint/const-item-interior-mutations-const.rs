//@ check-pass
//@ run-rustfix

#![allow(deprecated)]
#![allow(dead_code)]
#![feature(lock_value_accessors)]
#![feature(once_cell_try_insert)]
#![feature(once_cell_try)]

use std::sync::{Condvar, LazyLock, Mutex, Once, OnceLock, RwLock};
use std::time::Duration;

fn mutex() {
    const A: Mutex<i32> = Mutex::new(0);

    let _a = A.set(1);
    //~^ WARN mutation of an interior mutable `const` item with call to `set`

    let _a = A.replace(2);
    //~^ WARN mutation of an interior mutable `const` item with call to `replace`

    drop(A.lock());
    //~^ WARN mutation of an interior mutable `const` item with call to `lock`

    drop(A.try_lock());
    //~^ WARN mutation of an interior mutable `const` item with call to `try_lock`

    let _a = A.clear_poison();
    //~^ WARN mutation of an interior mutable `const` item with call to `clear_poison`
}

fn once() {
    const A: Once = Once::new();

    let _a = A.call_once(|| {});
    //~^ WARN mutation of an interior mutable `const` item with call to `call_once`

    let _a = A.call_once_force(|_| {});
    //~^ WARN mutation of an interior mutable `const` item with call to `call_once_force`

    let _a = A.wait();
    //~^ WARN mutation of an interior mutable `const` item with call to `wait`

    let _a = A.wait_force();
    //~^ WARN mutation of an interior mutable `const` item with call to `wait_force`
}

fn rwlock() {
    const A: RwLock<i32> = RwLock::new(0);

    let _a = A.set(1);
    //~^ WARN mutation of an interior mutable `const` item with call to `set`

    let _a = A.replace(2);
    //~^ WARN mutation of an interior mutable `const` item with call to `replace`

    drop(A.read());
    //~^ WARN mutation of an interior mutable `const` item with call to `read`

    drop(A.try_read());
    //~^ WARN mutation of an interior mutable `const` item with call to `try_read`

    drop(A.write());
    //~^ WARN mutation of an interior mutable `const` item with call to `write`

    drop(A.try_write());
    //~^ WARN mutation of an interior mutable `const` item with call to `try_write`
}

fn lazy_lock() {
    const A: LazyLock<i32> = LazyLock::new(|| 0);

    let _a = LazyLock::force(&A);
    //~^ WARN mutation of an interior mutable `const` item with call to `force`

    let _a = LazyLock::get(&A);
    //~^ WARN mutation of an interior mutable `const` item with call to `get`
}

fn once_lock() {
    const A: OnceLock<i32> = OnceLock::new();

    let _a = A.get();
    //~^ WARN mutation of an interior mutable `const` item with call to `get`

    let _a = A.wait();
    //~^ WARN mutation of an interior mutable `const` item with call to `wait`

    let _a = A.set(10);
    //~^ WARN mutation of an interior mutable `const` item with call to `set`

    let _a = A.try_insert(20);
    //~^ WARN mutation of an interior mutable `const` item with call to `try_insert`

    let _a = A.get_or_init(|| 30);
    //~^ WARN mutation of an interior mutable `const` item with call to `get_or_init`

    let _a = A.get_or_try_init(|| Ok::<_, ()>(40));
    //~^ WARN mutation of an interior mutable `const` item with call to `get_or_try_init`
}

fn condvar() {
    const A: Condvar = Condvar::new();

    let mutex = Mutex::new(0);
    let guard = mutex.lock().unwrap();

    let _a = A.wait(guard);
    //~^ WARN mutation of an interior mutable `const` item with call to `wait`

    let mutex = Mutex::new(0);
    let guard = mutex.lock().unwrap();
    let _a = A.wait_while(guard, |x| *x == 0);
    //~^ WARN mutation of an interior mutable `const` item with call to `wait_while`

    let mutex = Mutex::new(0);
    let guard = mutex.lock().unwrap();
    let _a = A.wait_timeout_ms(guard, 10);
    //~^ WARN mutation of an interior mutable `const` item with call to `wait_timeout_ms`

    let mutex = Mutex::new(0);
    let guard = mutex.lock().unwrap();
    let _a = A.wait_timeout(guard, Duration::from_millis(10));
    //~^ WARN mutation of an interior mutable `const` item with call to `wait_timeout`

    let mutex = Mutex::new(0);
    let guard = mutex.lock().unwrap();
    let _a = A.wait_timeout_while(guard, Duration::from_millis(10), |x| *x == 0);
    //~^ WARN mutation of an interior mutable `const` item with call to `wait_timeout_while`

    let _a = A.notify_one();
    //~^ WARN mutation of an interior mutable `const` item with call to `notify_one`

    let _a = A.notify_all();
    //~^ WARN mutation of an interior mutable `const` item with call to `notify_all`
}

fn main() {}
