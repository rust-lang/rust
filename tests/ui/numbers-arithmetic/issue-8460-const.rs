// revisions: noopt opt opt_with_overflow_checks
//[noopt]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

use std::thread;

fn main() {
    assert!(
        thread::spawn(move || {
            isize::MIN / -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i8::MIN / -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i16::MIN / -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i32::MIN / -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i64::MIN / -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i128::MIN / -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1isize / 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i8 / 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i16 / 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i32 / 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i64 / 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i128 / 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            isize::MIN % -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i8::MIN % -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i16::MIN % -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i32::MIN % -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i64::MIN % -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            i128::MIN % -1; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1isize % 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i8 % 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i16 % 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i32 % 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i64 % 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
    assert!(
        thread::spawn(move || {
            1i128 % 0; //~ ERROR operation will panic
        })
        .join()
        .is_err()
    );
}
