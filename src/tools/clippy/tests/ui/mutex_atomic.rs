#![warn(clippy::mutex_integer)]
#![warn(clippy::mutex_atomic)]
#![allow(clippy::borrow_as_ptr)]

fn main() {
    use std::sync::Mutex;
    Mutex::new(true);
    //~^ mutex_atomic

    Mutex::new(5usize);
    //~^ mutex_atomic

    Mutex::new(9isize);
    //~^ mutex_atomic

    let mut x = 4u32;
    Mutex::new(&x as *const u32);
    //~^ mutex_atomic

    Mutex::new(&mut x as *mut u32);
    //~^ mutex_atomic

    Mutex::new(0u32);
    //~^ mutex_integer

    Mutex::new(0i32);
    //~^ mutex_integer

    Mutex::new(0f32); // there are no float atomics, so this should not lint
    Mutex::new(0u8);
    //~^ mutex_integer

    Mutex::new(0i16);
    //~^ mutex_integer

    let _x: Mutex<i8> = Mutex::new(0);
    //~^ mutex_integer

    const X: i64 = 0;
    Mutex::new(X);
    //~^ mutex_integer

    // there are no 128 atomics, so these two should not lint
    {
        Mutex::new(0u128);
        let _x: Mutex<i128> = Mutex::new(0);
    }
}
