// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "staticlib"]
#![feature(c_variadic)]
#![feature(libc)]

extern crate libc;

use libc::{c_char, c_double, c_int, c_long, c_longlong};
use std::ffi::VaList;
use std::slice;
use std::ffi::CStr;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum AnswerType {
    Double,
    Long,
    LongLong,
    Int,
    Byte,
    CStr,
    Skip,
}

#[repr(C)]
pub union AnswerData {
    pub double: c_double,
    pub long: c_long,
    pub longlong: c_longlong,
    pub int: c_int,
    pub byte: c_char,
    pub cstr: *const c_char,
    pub skip_ty: AnswerType,
}

#[repr(C)]
pub struct Answer {
    tag: AnswerType,
    data: AnswerData,
}

#[no_mangle]
pub unsafe fn compare_answers(answers: &[Answer], mut ap: VaList) -> usize {
    for (i, answer) in answers.iter().enumerate() {
        match answer {
            Answer { tag: AnswerType::Double, data: AnswerData { double: d } } => {
                let tmp = ap.arg::<c_double>();
                if d.floor() != tmp.floor() {
                    println!("Double: {} != {}", d, tmp);
                    return i + 1;
                }
            }
            Answer { tag: AnswerType::Long, data: AnswerData { long: l } } => {
                let tmp =  ap.arg::<c_long>();
                if *l != tmp {
                    println!("Long: {} != {}", l, tmp);
                    return i + 1;
                }
            }
            Answer { tag: AnswerType::LongLong, data: AnswerData { longlong: l } } => {
                let tmp =  ap.arg::<c_longlong>();
                if *l != tmp {
                    println!("Long Long: {} != {}", l, tmp);
                    return i + 1;
                }
            }
            Answer { tag: AnswerType::Int, data: AnswerData { int: n } } => {
                let tmp = ap.arg::<c_int>();
                if *n != tmp {
                    println!("Int: {} != {}", n, tmp);
                    return i + 1;
                }
            }
            Answer { tag: AnswerType::Byte, data: AnswerData { byte: b } } => {
                let tmp = ap.arg::<c_char>();
                if *b != tmp {
                    println!("Byte: {} != {}", b, tmp);
                    return i + 1;
                }
            }
            Answer { tag: AnswerType::CStr, data: AnswerData { cstr: c0 } } => {
                let c1 = ap.arg::<*const c_char>();
                let cstr0 = CStr::from_ptr(*c0);
                let cstr1 = CStr::from_ptr(c1);
                if cstr0 != cstr1 {
                    println!("C String: {:?} != {:?}", cstr0, cstr1);
                    return i + 1;
                }
            }
            _ => {
                println!("Unknown type!");
                return i + 1;
            }
        }
    }
    return 0;
}

#[no_mangle]
pub unsafe extern "C" fn check_rust(argc: usize, answers: *const Answer, ap: VaList) -> usize {
    let slice = slice::from_raw_parts(answers, argc);
    compare_answers(slice, ap)
}

#[no_mangle]
pub unsafe extern "C" fn check_rust_copy(argc: usize, answers: *const Answer,
                                         mut ap: VaList) -> usize {
    let slice = slice::from_raw_parts(answers, argc);
    let mut skip_n = 0;
    for (i, answer) in slice.iter().enumerate() {
        match answer {
            Answer { tag: AnswerType::Skip, data: AnswerData { skip_ty } } => {
                match skip_ty {
                    AnswerType::Double => { ap.arg::<c_double>(); }
                    AnswerType::Long => { ap.arg::<c_long>(); }
                    AnswerType::LongLong => { ap.arg::<c_longlong>(); }
                    AnswerType::Int => { ap.arg::<c_int>(); }
                    AnswerType::Byte => { ap.arg::<c_char>(); }
                    AnswerType::CStr => { ap.arg::<*const c_char>(); }
                    _ => { return i; }
                };
            }
            _ => {
                skip_n = i;
                break;
            }
        }
    }

    ap.copy(|ap| {
        compare_answers(&slice[skip_n..], ap)
    })
}
