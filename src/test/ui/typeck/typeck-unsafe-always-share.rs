// Verify that UnsafeCell is *always* !Sync regardless if `T` is sync.

#![feature(optin_builtin_traits)]

use std::cell::UnsafeCell;
use std::marker::Sync;

struct MySync<T> {
    u: UnsafeCell<T>
}

struct NoSync;
impl !Sync for NoSync {}

fn test<T: Sync>(s: T) {}

fn main() {
    let us = UnsafeCell::new(MySync{u: UnsafeCell::new(0)});
    test(us);
    //~^ ERROR `std::cell::UnsafeCell<MySync<{integer}>>` cannot be shared between threads safely

    let uns = UnsafeCell::new(NoSync);
    test(uns);
    //~^ ERROR `std::cell::UnsafeCell<NoSync>` cannot be shared between threads safely [E0277]

    let ms = MySync{u: uns};
    test(ms);
    //~^ ERROR `std::cell::UnsafeCell<NoSync>` cannot be shared between threads safely [E0277]

    test(NoSync);
    //~^ ERROR `NoSync` cannot be shared between threads safely [E0277]
}
