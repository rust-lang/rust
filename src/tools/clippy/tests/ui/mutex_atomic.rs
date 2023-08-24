#![warn(clippy::all)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::mutex_atomic)]
#![allow(clippy::borrow_as_ptr)]

fn main() {
    use std::sync::Mutex;
    Mutex::new(true);
    //~^ ERROR: consider using an `AtomicBool` instead of a `Mutex` here; if you just want
    //~| NOTE: `-D clippy::mutex-atomic` implied by `-D warnings`
    Mutex::new(5usize);
    //~^ ERROR: consider using an `AtomicUsize` instead of a `Mutex` here; if you just wan
    Mutex::new(9isize);
    //~^ ERROR: consider using an `AtomicIsize` instead of a `Mutex` here; if you just wan
    let mut x = 4u32;
    Mutex::new(&x as *const u32);
    //~^ ERROR: consider using an `AtomicPtr` instead of a `Mutex` here; if you just want
    Mutex::new(&mut x as *mut u32);
    //~^ ERROR: consider using an `AtomicPtr` instead of a `Mutex` here; if you just want
    Mutex::new(0u32);
    //~^ ERROR: consider using an `AtomicUsize` instead of a `Mutex` here; if you just wan
    //~| NOTE: `-D clippy::mutex-integer` implied by `-D warnings`
    Mutex::new(0i32);
    //~^ ERROR: consider using an `AtomicIsize` instead of a `Mutex` here; if you just wan
    Mutex::new(0f32); // there are no float atomics, so this should not lint
}
