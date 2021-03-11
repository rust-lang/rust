//! The field `guard` is never used directly, but it is still useful for its side effect when
//! dropped. Since rustc doesn't consider a `Drop` impl as a use, we want to make sure we at least
//! produce a helpful diagnostic that points the user to what they can do if they indeed intended to
//! have a field that is only used for its `Drop` side effect.
//!
//! Issue: https://github.com/rust-lang/rust/issues/81658

#![deny(dead_code)]

use std::sync::{Mutex, MutexGuard};

/// Holds a locked value until it is dropped
pub struct Locked<'a, T> {
    // Field is kept for its affect when dropped, but otherwise unused
    guard: MutexGuard<'a, T>, //~ ERROR field is never read
}

impl<'a, T> Locked<'a, T> {
    pub fn new(value: &'a Mutex<T>) -> Self {
        Self {
            guard: value.lock().unwrap(),
        }
    }
}

fn main() {
    let items = Mutex::new(vec![1, 2, 3]);

    // Hold a lock on items while doing something else
    let result = {
        // The lock will be released at the end of this scope
        let _lock = Locked::new(&items);

        do_something_else()
    };

    println!("{}", result);
}

fn do_something_else() -> i32 {
    1 + 1
}
