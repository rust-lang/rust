// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(align_offset)]

#[derive(Clone, Copy)]
#[repr(packed)]
struct A3(u16, u8);
struct A4(u32);
#[repr(packed)]
struct A5(u32, u8);
#[repr(packed)]
struct A6(u32, u16);
#[repr(packed)]
struct A7(u32, u16, u8);
#[repr(packed)]
struct A8(u32, u32);
#[repr(packed)]
struct A9(u32, u32, u8);
#[repr(packed)]
struct A10(u32, u32, u16);

unsafe fn test_weird_stride<T>(ptr: *const T, align: usize) -> bool {
    let numptr = ptr as usize;
    let mut expected = usize::max_value();
    // Naive but definitely correct way to find the *first* aligned element of stride::<T>.
    for el in (0..align) {
        if (numptr + el * ::std::mem::size_of::<T>()) % align == 0 {
            expected = el;
            break;
        }
    }
    let got = ptr.align_offset(align);
    if got != expected {
        eprintln!("aligning {:p} (with stride of {}) to {}, expected {}, got {}", ptr, ::std::mem::size_of::<T>(), align, expected, got);
        return true;
    }
    return false;
}

fn main() {
    unsafe {
        // For pointers of stride = 0, the pointer is already aligned or it cannot be aligned at
        // all, because no amount of elements will align the pointer.
        let mut p = 1;
        while p < 1024 {
            assert_eq!((p as *const ()).align_offset(p), 0);
            if (p != 1) {
                assert_eq!(((p + 1) as *const ()).align_offset(p), !0);
            }
            p = (p + 1).next_power_of_two();
        }

        // For pointers of stride = 1, the pointer can always be aligned. The offset is equal to
        // number of bytes.
        let mut align = 1;
        while align < 1024 {
            for ptr in 1..2*align {
                let expected = ptr % align;
                let offset = if expected == 0 { 0 } else { align - expected };
                assert_eq!((ptr as *const u8).align_offset(align), offset,
                           "ptr = {}, align = {}, size = 1", ptr, align);
                align = (align + 1).next_power_of_two();
            }
        }


        // For pointers of stride != 1, we verify the algorithm against the naivest possible
        // implementation
        let mut align = 1;
        let mut x = false;
        while align < 1024 {
            for ptr in 1usize..4*align {
                x |= test_weird_stride::<A3>(ptr as *const A3, align);
                x |= test_weird_stride::<A4>(ptr as *const A4, align);
                x |= test_weird_stride::<A5>(ptr as *const A5, align);
                x |= test_weird_stride::<A6>(ptr as *const A6, align);
                x |= test_weird_stride::<A7>(ptr as *const A7, align);
                x |= test_weird_stride::<A8>(ptr as *const A8, align);
                x |= test_weird_stride::<A9>(ptr as *const A9, align);
                x |= test_weird_stride::<A10>(ptr as *const A10, align);
            }
            align = (align + 1).next_power_of_two();
        }
        assert!(!x);
    }
}
