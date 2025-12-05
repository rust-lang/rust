//@no-rustfix
#![warn(clippy::mutex_atomic, clippy::mutex_integer)]

use std::sync::Mutex;

fn issue13378() {
    static MTX: Mutex<u32> = Mutex::new(0);
    //~^ mutex_integer

    // unfixable because we don't fix this `lock`
    let mut guard = MTX.lock().unwrap();
    *guard += 1;
}
