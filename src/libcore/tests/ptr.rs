// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::ptr::*;
use core::cell::RefCell;

#[test]
fn test() {
    unsafe {
        struct Pair {
            fst: isize,
            snd: isize
        };
        let mut p = Pair {fst: 10, snd: 20};
        let pptr: *mut Pair = &mut p;
        let iptr: *mut isize = pptr as *mut isize;
        assert_eq!(*iptr, 10);
        *iptr = 30;
        assert_eq!(*iptr, 30);
        assert_eq!(p.fst, 30);

        *pptr = Pair {fst: 50, snd: 60};
        assert_eq!(*iptr, 50);
        assert_eq!(p.fst, 50);
        assert_eq!(p.snd, 60);

        let v0 = vec![32000u16, 32001u16, 32002u16];
        let mut v1 = vec![0u16, 0u16, 0u16];

        copy(v0.as_ptr().offset(1), v1.as_mut_ptr().offset(1), 1);
        assert!((v1[0] == 0u16 &&
                 v1[1] == 32001u16 &&
                 v1[2] == 0u16));
        copy(v0.as_ptr().offset(2), v1.as_mut_ptr(), 1);
        assert!((v1[0] == 32002u16 &&
                 v1[1] == 32001u16 &&
                 v1[2] == 0u16));
        copy(v0.as_ptr(), v1.as_mut_ptr().offset(2), 1);
        assert!((v1[0] == 32002u16 &&
                 v1[1] == 32001u16 &&
                 v1[2] == 32000u16));
    }
}

#[test]
fn test_is_null() {
    let p: *const isize = null();
    assert!(p.is_null());

    let q = unsafe { p.offset(1) };
    assert!(!q.is_null());

    let mp: *mut isize = null_mut();
    assert!(mp.is_null());

    let mq = unsafe { mp.offset(1) };
    assert!(!mq.is_null());
}

#[test]
fn test_as_ref() {
    unsafe {
        let p: *const isize = null();
        assert_eq!(p.as_ref(), None);

        let q: *const isize = &2;
        assert_eq!(q.as_ref().unwrap(), &2);

        let p: *mut isize = null_mut();
        assert_eq!(p.as_ref(), None);

        let q: *mut isize = &mut 2;
        assert_eq!(q.as_ref().unwrap(), &2);

        // Lifetime inference
        let u = 2isize;
        {
            let p = &u as *const isize;
            assert_eq!(p.as_ref().unwrap(), &2);
        }
    }
}

#[test]
fn test_as_mut() {
    unsafe {
        let p: *mut isize = null_mut();
        assert!(p.as_mut() == None);

        let q: *mut isize = &mut 2;
        assert!(q.as_mut().unwrap() == &mut 2);

        // Lifetime inference
        let mut u = 2isize;
        {
            let p = &mut u as *mut isize;
            assert!(p.as_mut().unwrap() == &mut 2);
        }
    }
}

#[test]
fn test_ptr_addition() {
    unsafe {
        let xs = vec![5; 16];
        let mut ptr = xs.as_ptr();
        let end = ptr.offset(16);

        while ptr < end {
            assert_eq!(*ptr, 5);
            ptr = ptr.offset(1);
        }

        let mut xs_mut = xs;
        let mut m_ptr = xs_mut.as_mut_ptr();
        let m_end = m_ptr.offset(16);

        while m_ptr < m_end {
            *m_ptr += 5;
            m_ptr = m_ptr.offset(1);
        }

        assert!(xs_mut == vec![10; 16]);
    }
}

#[test]
fn test_ptr_subtraction() {
    unsafe {
        let xs = vec![0,1,2,3,4,5,6,7,8,9];
        let mut idx = 9;
        let ptr = xs.as_ptr();

        while idx >= 0 {
            assert_eq!(*(ptr.offset(idx as isize)), idx as isize);
            idx = idx - 1;
        }

        let mut xs_mut = xs;
        let m_start = xs_mut.as_mut_ptr();
        let mut m_ptr = m_start.offset(9);

        while m_ptr >= m_start {
            *m_ptr += *m_ptr;
            m_ptr = m_ptr.offset(-1);
        }

        assert_eq!(xs_mut, [0,2,4,6,8,10,12,14,16,18]);
    }
}

#[test]
fn test_set_memory() {
    let mut xs = [0u8; 20];
    let ptr = xs.as_mut_ptr();
    unsafe { write_bytes(ptr, 5u8, xs.len()); }
    assert!(xs == [5u8; 20]);
}

#[test]
fn test_unsized_unique() {
    let xs: &mut [i32] = &mut [1, 2, 3];
    let ptr = unsafe { Unique::new(xs as *mut [i32]) };
    let ys = unsafe { &mut **ptr };
    let zs: &mut [i32] = &mut [1, 2, 3];
    assert!(ys == zs);
}

#[test]
#[allow(warnings)]
// Have a symbol for the test below. It doesn’t need to be an actual variadic function, match the
// ABI, or even point to an actual executable code, because the function itself is never invoked.
#[no_mangle]
pub fn test_variadic_fnptr() {
    use core::hash::{Hash, SipHasher};
    extern {
        fn test_variadic_fnptr(_: u64, ...) -> f64;
    }
    let p: unsafe extern fn(u64, ...) -> f64 = test_variadic_fnptr;
    let q = p.clone();
    assert_eq!(p, q);
    assert!(!(p < q));
    let mut s = SipHasher::new();
    assert_eq!(p.hash(&mut s), q.hash(&mut s));
}

#[test]
fn write_unaligned_drop() {
    thread_local! {
        static DROPS: RefCell<Vec<u32>> = RefCell::new(Vec::new());
    }

    struct Dropper(u32);

    impl Drop for Dropper {
        fn drop(&mut self) {
            DROPS.with(|d| d.borrow_mut().push(self.0));
        }
    }

    {
        let c = Dropper(0);
        let mut t = Dropper(1);
        unsafe { write_unaligned(&mut t, c); }
    }
    DROPS.with(|d| assert_eq!(*d.borrow(), [0]));
}
