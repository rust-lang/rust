#![deny(const_err)]

use std::{isize, i8, i16, i32, i64};
use std::thread;

fn main() {
    assert!(thread::spawn(move|| { isize::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { i8::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { i16::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { i32::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { i64::MIN / -1; }).join().is_err());
    //~^ ERROR attempt to divide with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1isize / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1i8 / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1i16 / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1i32 / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1i64 / 0; }).join().is_err());
    //~^ ERROR attempt to divide by zero
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { isize::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { i8::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { i16::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { i32::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { i64::MIN % -1; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with overflow
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1isize % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1i8 % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1i16 % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1i32 % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
    //~| ERROR this expression will panic at runtime
    assert!(thread::spawn(move|| { 1i64 % 0; }).join().is_err());
    //~^ ERROR attempt to calculate the remainder with a divisor of zero
    //~| ERROR this expression will panic at runtime
}
