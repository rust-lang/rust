// revisions: noopt opt opt_with_overflow_checks
//[noopt]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

// build-fail

#![deny(const_err)]

use std::thread;

fn main() {
    assert!(thread::spawn(move|| { isize::MIN / -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i8::MIN / -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i16::MIN / -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i32::MIN / -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i64::MIN / -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i128::MIN / -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { 1isize / 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i8 / 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i16 / 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i32 / 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i64 / 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i128 / 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { isize::MIN % -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i8::MIN % -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i16::MIN % -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i32::MIN % -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i64::MIN % -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { i128::MIN % -1; }).join().is_err());
    //~^ ERROR arithmetic operation will overflow
    assert!(thread::spawn(move|| { 1isize % 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i8 % 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i16 % 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i32 % 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i64 % 0; }).join().is_err());
    //~^ ERROR operation will panic
    assert!(thread::spawn(move|| { 1i128 % 0; }).join().is_err());
    //~^ ERROR operation will panic
}
