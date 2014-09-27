// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Safe thread-local storage library

// FIXME: #17572: add support for TLS variables with destructors

#![crate_name = "tls"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/")]

#![no_std]
#![experimental]
#![feature(phase, macro_rules)]

#[phase(plugin, link)]
extern crate core;

// Allow testing this library

#[cfg(test)] extern crate debug;
#[cfg(test)] extern crate native;
#[cfg(test)] #[phase(plugin, link)] extern crate std;
#[cfg(test)] #[phase(plugin, link)] extern crate log;

/// Thread local Cell with a constant initializer.
#[macro_export]
macro_rules! tls_cell(
    ($name:ident, $t:ty, $init:expr) => {
        #[allow(dead_code)]
        mod $name {
            #[thread_local]
            static mut VALUE: $t = $init;

            #[inline(always)]
            pub fn set(value: $t) {
                unsafe {
                    VALUE = value;
                }
            }

            #[inline(always)]
            pub fn get() -> $t {
                unsafe {
                    VALUE
                }
            }
        }
    }
)

/// Thread local Cell with a dynamic initializer.
#[macro_export]
macro_rules! tls_cell_dynamic(
    ($name:ident, $t:ty, $init:expr) => {
        #[allow(dead_code)]
        mod $name {
            use core::option::{Option, None, Some};

            #[thread_local]
            static mut VALUE: Option<$t> = None;

            #[inline]
            fn init() {
                if unsafe { VALUE.is_none() } {
                    let tmp = $init;
                    unsafe { VALUE = Some(tmp) }
                }
            }

            #[inline]
            pub fn set(value: $t) {
                init();
                unsafe {
                    VALUE = Some(value);
                }
            }

            #[inline]
            pub fn get() -> $t {
                init();
                unsafe {
                    VALUE.unwrap()
                }
            }
        }
    }
)

/// Thread local RefCell with a constant initializer.
#[macro_export]
macro_rules! tls_refcell(
    ($name:ident, $t:ty, $init:expr) => {
        #[allow(dead_code)]
        mod $name {
            use core::cell::{RefCell, Ref, RefMut, UnsafeCell};
            use core::kinds::marker;
            use core::option::Option;

            // no way to ignore privacy
            type BorrowFlag = uint;
            static UNUSED: BorrowFlag = 0;

            // no way to ignore privacy
            pub struct NotCell<T> {
                value: UnsafeCell<T>,
                noshare: marker::NoSync,
            }

            // no way to ignore privacy
            struct NotRefCell<T> {
                value: UnsafeCell<T>,
                borrow: NotCell<BorrowFlag>,
                nocopy: marker::NoCopy,
                noshare: marker::NoSync,
            }

            // cannot call RefCell::new in a constant expression
            #[thread_local]
            static mut VALUE: NotRefCell<$t> = NotRefCell {
                value: UnsafeCell { value: $init },
                borrow: NotCell {
                                    value: UnsafeCell { value: UNUSED },
                                    noshare: marker::NoSync
                                },
                nocopy: marker::NoCopy,
                noshare: marker::NoSync
            };

            #[inline]
            pub fn try_borrow() -> Option<Ref<'static, $t>> {
                unsafe {
                    let ptr: &RefCell<$t> = ::core::mem::transmute(&VALUE);
                    ptr.try_borrow()
                }
            }

            #[inline]
            pub fn borrow() -> Ref<'static, $t> {
                unsafe {
                    let ptr: &RefCell<$t> = ::core::mem::transmute(&VALUE);
                    ptr.borrow()
                }
            }

            #[inline]
            pub fn try_borrow_mut() -> Option<RefMut<'static, $t>> {
                unsafe {
                    let ptr: &mut RefCell<$t> = ::core::mem::transmute(&mut VALUE);
                    ptr.try_borrow_mut()
                }
            }

            #[inline]
            pub fn borrow_mut() -> RefMut<'static, $t> {
                unsafe {
                    let ptr: &mut RefCell<$t> = ::core::mem::transmute(&mut VALUE);
                    ptr.borrow_mut()
                }
            }
        }
    }
)

/// Thread local RefCell with a dynamic initializer.
#[macro_export]
macro_rules! tls_refcell_dynamic(
    ($name:ident, $t:ty, $init:expr) => {
        #[allow(dead_code)]
        mod $name {
            use core::cell::{RefCell, Ref, RefMut};
            use core::option::{Option, None, Some};

            #[thread_local]
            static mut VALUE: Option<RefCell<$t>> = None;

            #[inline]
            fn init() {
                if unsafe { VALUE.is_none() } {
                    let tmp = $init;
                    unsafe { VALUE = Some(RefCell::new(tmp)) }
                }
            }

            #[inline]
            pub fn try_borrow() -> Option<Ref<'static, $t>> {
                init();
                unsafe {
                    VALUE.as_ref().unwrap().try_borrow()
                }
            }

            #[inline]
            pub fn borrow() -> Ref<'static, $t> {
                init();
                unsafe {
                    VALUE.as_ref().unwrap().borrow()
                }
            }

            #[inline]
            pub fn try_borrow_mut() -> Option<RefMut<'static, $t>> {
                init();
                unsafe {
                    VALUE.as_mut().unwrap().try_borrow_mut()
                }
            }

            #[inline]
            pub fn borrow_mut() -> RefMut<'static, $t> {
                init();
                unsafe {
                    VALUE.as_mut().unwrap().borrow_mut()
                }
            }
        }
    }
)

// FIXME: #17579: need a fallback path on iOS and Android for now
#[cfg(all(test, not(target_os = "android"), not(target_os = "ios")))]
mod tests {
    use core::iter::range;
    use std::task::spawn;

    fn five() -> u32 {
        5
    }

    #[test]
    fn basic_tls_cell() {
        tls_cell!(a, u32, 5)

        for _ in range(0, 10u) {
            spawn(proc() {
                assert_eq!(a::get(), 5);
                a::set(10);
                assert_eq!(a::get(), 10);
            });
        }
    }

    #[test]
    fn basic_tls_cell_dynamic() {
        tls_cell_dynamic!(b, u32, super::five())

        for _ in range(0, 10u) {
            spawn(proc() {
                assert_eq!(b::get(), 5);
                b::set(10);
                assert_eq!(b::get(), 10);
            });
        }
    }

    #[test]
    fn basic_tls_refcell() {
        tls_refcell!(c, u32, 5)

        for _ in range(0, 10u) {
            spawn(proc() {
                assert_eq!(*c::borrow(), 5);
                *c::borrow_mut() = 10;
                assert_eq!(*c::borrow(), 10);
            });
        }
    }

    #[test]
    fn basic_tls_refcell_dynamic() {
        tls_refcell_dynamic!(d, u32, super::five())

        for _ in range(0, 10u) {
            spawn(proc() {
                assert_eq!(*d::borrow(), 5);
                *d::borrow_mut() = 10;
                assert_eq!(*d::borrow(), 10);
            });
        }
    }
}
