// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(deprecated)]
use core::ptr::*;
use libc::c_char;
use core::mem;
use libc;
use std::c_str::CString;

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
        assert!((*v1.get(0) == 0u16 &&
                 *v1.get(1) == 32001u16 &&
                 *v1.get(2) == 0u16));
        copy_memory(v1.as_mut_ptr(),
                    v0.as_ptr().offset(2), 1);
        assert!((*v1.get(0) == 32002u16 &&
                 *v1.get(1) == 32001u16 &&
                 *v1.get(2) == 0u16));
        copy_memory(v1.as_mut_ptr().offset(2),
                    v0.as_ptr(), 1u);
        assert!((*v1.get(0) == 32002u16 &&
                 *v1.get(1) == 32001u16 &&
                 *v1.get(2) == 32000u16));
    }
}

#[test]
fn test_position() {
    use libc::c_char;

    "hello".with_c_str(|p| {
        unsafe {
            assert!(2u == position(p, |c| *c == 'l' as c_char));
            assert!(4u == position(p, |c| *c == 'o' as c_char));
            assert!(5u == position(p, |c| *c == 0 as c_char));
        }
    })
}

#[test]
fn test_buf_len() {
    "hello".with_c_str(|p0| {
        "there".with_c_str(|p1| {
            "thing".with_c_str(|p2| {
                let v = vec![p0, p1, p2, null()];
                unsafe {
                    assert_eq!(buf_len(v.as_ptr()), 3u);
                }
            })
        })
    })
}

#[test]
fn test_is_null() {
    let p: *const int = null();
    assert!(p.is_null());
    assert!(!p.is_not_null());

    let q = unsafe { p.offset(1) };
    assert!(!q.is_null());
    assert!(q.is_not_null());

    let mp: *mut int = mut_null();
    assert!(mp.is_null());
    assert!(!mp.is_not_null());

    let mq = unsafe { mp.offset(1) };
    assert!(!mq.is_null());
    assert!(mq.is_not_null());
}

#[test]
fn test_to_option() {
    unsafe {
        let p: *const int = null();
        assert_eq!(p.to_option(), None);

        let q: *const int = &2;
        assert_eq!(q.to_option().unwrap(), &2);

        let p: *mut int = mut_null();
        assert_eq!(p.to_option(), None);

        let q: *mut int = &mut 2;
        assert_eq!(q.to_option().unwrap(), &2);
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
fn test_ptr_array_each_with_len() {
    unsafe {
        let one = "oneOne".to_c_str();
        let two = "twoTwo".to_c_str();
        let three = "threeThree".to_c_str();
        let arr = vec![
            one.as_ptr(),
            two.as_ptr(),
            three.as_ptr()
        ];
        let expected_arr = [
            one, two, three
        ];

        let mut ctr = 0;
        let mut iteration_count = 0;
        array_each_with_len(arr.as_ptr(), arr.len(), |e| {
                let actual = CString::new(e, false);
                assert_eq!(actual.as_str(), expected_arr[ctr].as_str());
                ctr += 1;
                iteration_count += 1;
            });
        assert_eq!(iteration_count, 3u);
    }
}

#[test]
fn test_ptr_array_each() {
    unsafe {
        let one = "oneOne".to_c_str();
        let two = "twoTwo".to_c_str();
        let three = "threeThree".to_c_str();
        let arr = vec![
            one.as_ptr(),
            two.as_ptr(),
            three.as_ptr(),
            // fake a null terminator
            null()
        ];
        let expected_arr = [
            one, two, three
        ];

        let arr_ptr = arr.as_ptr();
        let mut ctr = 0u;
        let mut iteration_count = 0u;
        array_each(arr_ptr, |e| {
                let actual = CString::new(e, false);
                assert_eq!(actual.as_str(), expected_arr[ctr].as_str());
                ctr += 1;
                iteration_count += 1;
            });
        assert_eq!(iteration_count, 3);
    }
}

#[test]
#[should_fail]
fn test_ptr_array_each_with_len_null_ptr() {
    unsafe {
        array_each_with_len(0 as *const *const libc::c_char, 1, |e| {
            CString::new(e, false).as_str().unwrap();
        });
    }
}
#[test]
#[should_fail]
fn test_ptr_array_each_null_ptr() {
    unsafe {
        array_each(0 as *const *const libc::c_char, |e| {
            CString::new(e, false).as_str().unwrap();
        });
    }
}

#[test]
fn test_set_memory() {
    let mut xs = [0u8, ..20];
    let ptr = xs.as_mut_ptr();
    unsafe { set_memory(ptr, 5u8, xs.len()); }
    assert!(xs == [5u8, ..20]);
}
