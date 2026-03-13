#![warn(clippy::unwrap_used)]
#![allow(clippy::unnecessary_literal_unwrap)]
#![allow(unused_variables)]
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(0);
    // This is allowed because `LockResult` is configured!
    let _guard = m.lock().unwrap();

    let optional: Option<i32> = Some(1);
    // This is not allowed!
    let _opt = optional.unwrap();
    //~^ ERROR: used `unwrap()` on an `Option` value

    let result: Result<i32, ()> = Ok(1);
    // This is not allowed!
    let _res = result.unwrap();
    //~^ ERROR: used `unwrap()` on a `Result` value
}
