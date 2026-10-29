//! Regression test for <https://github.com/rust-lang/rust/issues/23433>.
//! Don't fail if we encounter a NonNull<T> where T is an unsized type
//@ run-pass

use std::ptr::NonNull;

fn main() {
    let mut a = [0u8; 5];
    let b: Option<NonNull<[u8]>> = Some(NonNull::from(&mut a));
    match b {
        Some(_) => println!("Got `Some`"),
        None => panic!("Unexpected `None`"),
    }
}
