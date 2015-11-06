// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(asm)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn read(ptr: &u32) -> u32 {
    let out: u32;
    unsafe {
        asm!("mov $1, $0" : "=r" (out) : "*m" (ptr));
    }
    out
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn write(ptr: &mut u32, val: u32) {
    unsafe {
        asm!("mov $1, $0" : "=*m" (ptr) : "r" (val));
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn replace(ptr: &mut u32, val: u32) -> u32 {
    let out: u32;
    unsafe {
        asm!("mov $0, $1; mov $2, $0" : "+*m" (ptr), "=&r" (out) : "r" (val));
    }
    out
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn main() {
    let a = 1;
    assert_eq!(read(&a), 1);
    let mut b = 2;
    write(&mut b, 3);
    assert_eq!(b, 3);
    let mut c = 4;
    assert_eq!(replace(&mut c, 5), 4);
    assert_eq!(c, 5);
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn main() {}
