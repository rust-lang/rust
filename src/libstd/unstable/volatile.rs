// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Volatile Module.
 *
 * The order of execution of code is not guaranteed: the compiler may reorder statements
 * and instructions to make things more efficient. However, when you're
 * working with memory that other devices/programs may also be using,
 * you need to be specific about how and when that memory is read and written.
 *
 * That's where Volatile memory comes into play. They tell the compiler not to move the
 * order in which the memory is read or written.
 *
 * E.g.,
 *
 * ```rust
 * fn main() {
 *     let mut i = 100;
 *
 *     while i == 100 {
 *         // `i` doesn't change in here, but something else (device, process)
 *         // might change it and the compiler wouldn't know.
 *     }
 * }
 * ```
 *
 * The compiler may optimize the loop condition out of the program and replace it
 * with a simple `while true { ... }` statement. This happens
 * because it thinks the variable doesn't change.
 *
 * Replacing this with a volatile typed variable fixes the issue and doesn't allow
 * the compiler to optimize it out.
 *
 * ```rust
 * # use std::unstable::volatile::Volatile;
 * fn main() {
 *     let mut i = Volatile::new(100);
 *
 *     while i.load() == 100 {
 *         // `i` doesn't change in here, but something else (device, process)
 *         // might change it and the compiler wouldn't know.
 *     }
 * }
 * ```
 *
 * http://en.wikipedia.org/wiki/Volatile_variable
 * http://llvm.org/docs/LangRef.html#volatile-memory-accesses
 *
 * This module uses [LLVM's volatile intrinsics][llvm].
 * [llvm]: http://llvm.org/docs/LangRef.html#volatile-memory-accesses
 */

use intrinsics;
use kinds::Pod;
use std::ty::Unsafe;

trait VolatileSafe {}

impl VolatileSafe for i32 {}
impl VolatileSafe for u32 {}


/// A generic type that enforces volatile reads and writes to its contents.
///
/// Examples:
///
/// ```rust
/// # use std::unstable::volatile::Volatile;
/// let v: Volatile<int> = Volatile::new(10);
/// ```
#[experimental]
pub struct Volatile<T> {
    priv data: Unsafe<T>
}

impl<T: VolatileSafe + Pod> Volatile<T> {
    pub fn new(val: T) -> Volatile<T> {
        let mut s = Volatile { data: Unsafe::new(val) };
        s.store(val);
        s
    }

    #[inline]
    pub fn store(&mut self, val: T) {
        unsafe {
            intrinsics::volatile_store(self.data.get(), val);
        }
    }

    #[inline]
    pub fn load(&self) -> T {
        unsafe {
            intrinsics::volatile_load(self.data.get() as *T)
        }
    }
}

#[cfg(test)]
mod test {
    #![allow(experimental)]
    use super::*;

    #[test]
    fn volatile_ptr_new() {
        unsafe {
            let v: *Volatile<int> = &Volatile::new(5);
            assert_eq!((*v).load(), 5);
        }
    }

    #[test]
    fn volatile_static_alloc() {
        static foo: bool = true;
        fn get_foo() -> Volatile<bool> { Volatile::new(foo) }
        assert_eq!(get_foo().load(), foo);
    }

    #[test]
    fn volatile_ptr_store_and_load() {
        unsafe {
            let mut v = ~Volatile::new(1);
            assert_eq!(v.load(), 1);

            v.store(6);

            assert_eq!(v.load(), 6);
        }
    }

    #[test]
    fn volatile_int_load() {
        let vint = Volatile::new(16);
        assert_eq!(vint.load(), 16);
    }

    #[test]
    fn volatile_int_store() {
        let mut vint = Volatile::new(20);
        vint.store(10);
        assert_eq!(vint.load(), 10);
    }

    #[test]
    fn volatile_static_int() {
        static t: Volatile<int> = Volatile { data: 55 };
        static v: Volatile<bool> = Volatile { data: true };
        assert_eq!(v.load(), true);
        assert_eq!(t.load(), 55);
    }

    #[test]
    fn volatile_bool_new_and_load() {
        let b = Volatile::new(true);
        assert_eq!(b.load(), true);
    }

    #[test]
    fn volatile_bool_store() {
        let mut b = Volatile::new(true);
        b.store(false);
        assert_eq!(b.load(), false);
    }
}
