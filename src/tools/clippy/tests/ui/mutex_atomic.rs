#![warn(clippy::all)]
#![warn(clippy::mutex_integer)]
#![warn(clippy::mutex_atomic)]
#![allow(clippy::borrow_as_ptr)]

fn main() {
    use std::sync::Mutex;
    Mutex::new(true);
    Mutex::new(5usize);
    Mutex::new(9isize);
    let mut x = 4u32;
    Mutex::new(&x as *const u32);
    Mutex::new(&mut x as *mut u32);
    Mutex::new(0u32);
    Mutex::new(0i32);
    Mutex::new(0f32); // there are no float atomics, so this should not lint
}
