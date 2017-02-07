#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]
#![deny(mutex_integer)]

fn main() {
    use std::sync::Mutex;
    Mutex::new(true); //~ERROR Consider using an AtomicBool instead of a Mutex here.
    Mutex::new(5usize); //~ERROR Consider using an AtomicUsize instead of a Mutex here.
    Mutex::new(9isize); //~ERROR Consider using an AtomicIsize instead of a Mutex here.
    let mut x = 4u32;
    Mutex::new(&x as *const u32); //~ERROR Consider using an AtomicPtr instead of a Mutex here.
    Mutex::new(&mut x as *mut u32); //~ERROR Consider using an AtomicPtr instead of a Mutex here.
    Mutex::new(0u32); //~ERROR Consider using an AtomicUsize instead of a Mutex here.
    Mutex::new(0i32); //~ERROR Consider using an AtomicIsize instead of a Mutex here.
    Mutex::new(0f32); // there are no float atomics, so this should not lint
}
