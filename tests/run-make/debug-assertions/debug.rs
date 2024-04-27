#![feature(rustc_attrs)]
#![deny(warnings)]

use std::env;
use std::thread;

fn main() {
    let should_fail = env::args().nth(1) == Some("bad".to_string());

    assert_eq!(thread::spawn(debug_assert_eq).join().is_err(), should_fail);
    assert_eq!(thread::spawn(debug_assert).join().is_err(), should_fail);
    assert_eq!(thread::spawn(overflow).join().is_err(), should_fail);
}

fn debug_assert_eq() {
    let mut hit1 = false;
    let mut hit2 = false;
    debug_assert_eq!({ hit1 = true; 1 }, { hit2 = true; 2 });
    assert!(!hit1);
    assert!(!hit2);
}

fn debug_assert() {
    let mut hit = false;
    debug_assert!({ hit = true; false });
    assert!(!hit);
}

fn overflow() {
    fn add(a: u8, b: u8) -> u8 { a + b }

    add(200u8, 200u8);
}
