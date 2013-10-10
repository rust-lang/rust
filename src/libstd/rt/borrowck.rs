// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cell::Cell;
use c_str::{ToCStr, CString};
use libc::{c_char, size_t};
use option::{Option, None, Some};
use ptr::RawPtr;
use rt::env;
use rt::local::Local;
use rt::task::Task;
use str::{OwnedStr, StrSlice};
use str;
use sys;
use uint;
use unstable::raw;
use vec::ImmutableVector;

pub static FROZEN_BIT: uint = 1 << (uint::bits - 1);
pub static MUT_BIT: uint = 1 << (uint::bits - 2);
static ALL_BITS: uint = FROZEN_BIT | MUT_BIT;

#[deriving(Eq)]
pub struct BorrowRecord {
    box: *mut raw::Box<()>,
    file: *c_char,
    line: size_t
}

fn try_take_task_borrow_list() -> Option<~[BorrowRecord]> {
    do Local::borrow |task: &mut Task| {
        task.borrow_list.take()
    }
}

fn swap_task_borrow_list(f: &fn(~[BorrowRecord]) -> ~[BorrowRecord]) {
    let borrows = match try_take_task_borrow_list() {
        Some(l) => l,
        None => ~[]
    };
    let borrows = f(borrows);
    let borrows = Cell::new(borrows);
    do Local::borrow |task: &mut Task| {
        task.borrow_list = Some(borrows.take());
    }
}

pub fn clear_task_borrow_list() {
    // pub because it is used by the box annihilator.
    let _ = try_take_task_borrow_list();
}

unsafe fn fail_borrowed(box: *mut raw::Box<()>, file: *c_char, line: size_t) {
    debug_borrow("fail_borrowed: ", box, 0, 0, file, line);

    match try_take_task_borrow_list() {
        None => { // not recording borrows
            let msg = "borrowed";
            do msg.with_c_str |msg_p| {
                sys::begin_unwind_(msg_p, file, line);
            }
        }
        Some(borrow_list) => { // recording borrows
            let mut msg = ~"borrowed";
            let mut sep = " at ";
            for entry in borrow_list.rev_iter() {
                if entry.box == box {
                    msg.push_str(sep);
                    let filename = str::raw::from_c_str(entry.file);
                    msg.push_str(filename);
                    msg.push_str(format!(":{}", entry.line));
                    sep = " and at ";
                }
            }
            do msg.with_c_str |msg_p| {
                sys::begin_unwind_(msg_p, file, line)
            }
        }
    }
}

/// Because this code is so perf. sensitive, use a static constant so that
/// debug printouts are compiled out most of the time.
static ENABLE_DEBUG: bool = false;

#[inline]
unsafe fn debug_borrow<T,P:RawPtr<T>>(tag: &'static str,
                                      p: P,
                                      old_bits: uint,
                                      new_bits: uint,
                                      filename: *c_char,
                                      line: size_t) {
    //! A useful debugging function that prints a pointer + tag + newline
    //! without allocating memory.

    if ENABLE_DEBUG && env::debug_borrow() {
        debug_borrow_slow(tag, p, old_bits, new_bits, filename, line);
    }

    unsafe fn debug_borrow_slow<T,P:RawPtr<T>>(tag: &'static str,
                                               p: P,
                                               old_bits: uint,
                                               new_bits: uint,
                                               filename: *c_char,
                                               line: size_t) {
        let filename = CString::new(filename, false);
        rterrln!("{}{:#x} {:x} {:x} {}:{}",
                 tag, p.to_uint(), old_bits, new_bits,
                 filename.as_str().unwrap(), line);
    }
}

#[inline]
pub unsafe fn borrow_as_imm(a: *u8, file: *c_char, line: size_t) -> uint {
    let a = a as *mut raw::Box<()>;
    let old_ref_count = (*a).ref_count;
    let new_ref_count = old_ref_count | FROZEN_BIT;

    debug_borrow("borrow_as_imm:", a, old_ref_count, new_ref_count, file, line);

    if (old_ref_count & MUT_BIT) != 0 {
        fail_borrowed(a, file, line);
    }

    (*a).ref_count = new_ref_count;

    old_ref_count
}

#[inline]
pub unsafe fn borrow_as_mut(a: *u8, file: *c_char, line: size_t) -> uint {
    let a = a as *mut raw::Box<()>;
    let old_ref_count = (*a).ref_count;
    let new_ref_count = old_ref_count | MUT_BIT | FROZEN_BIT;

    debug_borrow("borrow_as_mut:", a, old_ref_count, new_ref_count, file, line);

    if (old_ref_count & (MUT_BIT|FROZEN_BIT)) != 0 {
        fail_borrowed(a, file, line);
    }

    (*a).ref_count = new_ref_count;

    old_ref_count
}

pub unsafe fn record_borrow(a: *u8, old_ref_count: uint,
                            file: *c_char, line: size_t) {
    if (old_ref_count & ALL_BITS) == 0 {
        // was not borrowed before
        let a = a as *mut raw::Box<()>;
        debug_borrow("record_borrow:", a, old_ref_count, 0, file, line);
        do swap_task_borrow_list |borrow_list| {
            let mut borrow_list = borrow_list;
            borrow_list.push(BorrowRecord {box: a, file: file, line: line});
            borrow_list
        }
    }
}

pub unsafe fn unrecord_borrow(a: *u8, old_ref_count: uint,
                              file: *c_char, line: size_t) {
    if (old_ref_count & ALL_BITS) == 0 {
        // was not borrowed before, so we should find the record at
        // the end of the list
        let a = a as *mut raw::Box<()>;
        debug_borrow("unrecord_borrow:", a, old_ref_count, 0, file, line);
        do swap_task_borrow_list |borrow_list| {
            let mut borrow_list = borrow_list;
            assert!(!borrow_list.is_empty());
            let br = borrow_list.pop();
            if br.box != a || br.file != file || br.line != line {
                let err = format!("wrong borrow found, br={:?}", br);
                do err.with_c_str |msg_p| {
                    sys::begin_unwind_(msg_p, file, line)
                }
            }
            borrow_list
        }
    }
}

#[inline]
pub unsafe fn return_to_mut(a: *u8, orig_ref_count: uint,
                            file: *c_char, line: size_t) {
    // Sometimes the box is null, if it is conditionally frozen.
    // See e.g. #4904.
    if !a.is_null() {
        let a = a as *mut raw::Box<()>;
        let old_ref_count = (*a).ref_count;
        let new_ref_count =
            (old_ref_count & !ALL_BITS) | (orig_ref_count & ALL_BITS);

        debug_borrow("return_to_mut:",
                     a, old_ref_count, new_ref_count, file, line);

        (*a).ref_count = new_ref_count;
    }
}

#[inline]
pub unsafe fn check_not_borrowed(a: *u8,
                                 file: *c_char,
                                 line: size_t) {
    let a = a as *mut raw::Box<()>;
    let ref_count = (*a).ref_count;
    debug_borrow("check_not_borrowed:", a, ref_count, 0, file, line);
    if (ref_count & FROZEN_BIT) != 0 {
        fail_borrowed(a, file, line);
    }
}
