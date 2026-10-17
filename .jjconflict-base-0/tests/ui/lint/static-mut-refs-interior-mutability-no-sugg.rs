//@ edition:2024
// Ensure that we don't make structured suggestions for interior mutability note
// when span is not available.

use std::sync::Mutex;

macro_rules! declare_mutex {
    () => {
        static mut MACRO_MUTEX: Mutex<bool> = Mutex::new(false);
    };
}

declare_mutex!();

fn main() {
    let _lock = unsafe { MACRO_MUTEX.lock().unwrap() };
    //~^ ERROR creating a shared reference to mutable static [static_mut_refs]
}
