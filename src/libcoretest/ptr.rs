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
use core::mem;

#[test]
fn test() {
    unsafe {
        struct Pair {
            fst: int,
            snd: int
        };
        let mut p = Pair {fst: 10, snd: 20};
        let pptr: *mut Pair = &mut p;
        let iptr: *mut int = mem::transmute(pptr);
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

        copy_memory(v1.as_mut_ptr().offset(1),
                    v0.as_ptr().offset(1), 1);
        assert!((v1[0] == 0u16 &&
                 v1[1] == 32001u16 &&
                 v1[2] == 0u16));
        copy_memory(v1.as_mut_ptr(),
                    v0.as_ptr().offset(2), 1);
        assert!((v1[0] == 32002u16 &&
                 v1[1] == 32001u16 &&
                 v1[2] == 0u16));
        copy_memory(v1.as_mut_ptr().offset(2),
                    v0.as_ptr(), 1u);
        assert!((v1[0] == 32002u16 &&
                 v1[1] == 32001u16 &&
                 v1[2] == 32000u16));
    }
}

#[test]
fn test_is_null() {
    let p: *const int = null();
    assert!(p.is_null());
    assert!(!p.is_not_null());

    let q = unsafe { p.offset(1) };
    assert!(!q.is_null());
    assert!(q.is_not_null());

    let mp: *mut int = null_mut();
    assert!(mp.is_null());
    assert!(!mp.is_not_null());

    let mq = unsafe { mp.offset(1) };
    assert!(!mq.is_null());
    assert!(mq.is_not_null());
}

#[test]
fn test_as_ref() {
    unsafe {
        let p: *const int = null();
        assert_eq!(p.as_ref(), None);

        let q: *const int = &2;
        assert_eq!(q.as_ref().unwrap(), &2);

        let p: *mut int = null_mut();
        assert_eq!(p.as_ref(), None);

        let q: *mut int = &mut 2;
        assert_eq!(q.as_ref().unwrap(), &2);

        // Lifetime inference
        let u = 2i;
        {
            let p: *const int = &u as *const _;
            assert_eq!(p.as_ref().unwrap(), &2);
        }
    }
}

#[test]
fn test_as_mut() {
    unsafe {
        let p: *mut int = null_mut();
        assert!(p.as_mut() == None);

        let q: *mut int = &mut 2;
        assert!(q.as_mut().unwrap() == &mut 2);

        // Lifetime inference
        let mut u = 2i;
        {
            let p: *mut int = &mut u as *mut _;
            assert!(p.as_mut().unwrap() == &mut 2);
        }
    }
}

#[test]
fn test_ptr_addition() {
    unsafe {
        let xs = Vec::from_elem(16, 5i);
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

        assert!(xs_mut == Vec::from_elem(16, 10i));
    }
}

#[test]
fn test_ptr_subtraction() {
    unsafe {
        let xs = vec![0,1,2,3,4,5,6,7,8,9];
        let mut idx = 9i8;
        let ptr = xs.as_ptr();

        while idx >= 0i8 {
            assert_eq!(*(ptr.offset(idx as int)), idx as int);
            idx = idx - 1i8;
        }

        let mut xs_mut = xs;
        let m_start = xs_mut.as_mut_ptr();
        let mut m_ptr = m_start.offset(9);

        while m_ptr >= m_start {
            *m_ptr += *m_ptr;
            m_ptr = m_ptr.offset(-1);
        }

        assert!(xs_mut == vec![0,2,4,6,8,10,12,14,16,18]);
    }
}

#[test]
fn test_set_memory() {
    let mut xs = [0u8, ..20];
    let ptr = xs.as_mut_ptr();
    unsafe { set_memory(ptr, 5u8, xs.len()); }
    assert!(xs == [5u8, ..20]);
}
