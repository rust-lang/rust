// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cast::transmute;
use libc::{c_char, c_void, size_t, STDERR_FILENO};
use io;
use io::{Writer, WriterUtil};
use option::{Option, None, Some};
use uint;
use str;
use str::{OwnedStr, StrSlice};
use sys;
use unstable::raw;
use vec::ImmutableVector;

#[allow(non_camel_case_types)]
type rust_task = c_void;

pub static FROZEN_BIT: uint = 1 << (uint::bits - 1);
pub static MUT_BIT: uint = 1 << (uint::bits - 2);
static ALL_BITS: uint = FROZEN_BIT | MUT_BIT;

#[deriving(Eq)]
struct BorrowRecord {
    box: *mut raw::Box<()>,
    file: *c_char,
    line: size_t
}

fn try_take_task_borrow_list() -> Option<~[BorrowRecord]> {
    unsafe {
        let cur_task: *rust_task = rust_try_get_task();
        if cur_task.is_not_null() {
            let ptr = rust_take_task_borrow_list(cur_task);
            if ptr.is_null() {
                None
            } else {
                let v: ~[BorrowRecord] = transmute(ptr);
                Some(v)
            }
        } else {
            None
        }
    }
}

fn swap_task_borrow_list(f: &fn(~[BorrowRecord]) -> ~[BorrowRecord]) {
    unsafe {
        let cur_task: *rust_task = rust_try_get_task();
        if cur_task.is_not_null() {
            let mut borrow_list: ~[BorrowRecord] = {
                let ptr = rust_take_task_borrow_list(cur_task);
                if ptr.is_null() { ~[] } else { transmute(ptr) }
            };
            borrow_list = f(borrow_list);
            rust_set_task_borrow_list(cur_task, transmute(borrow_list));
        }
    }
}

pub unsafe fn clear_task_borrow_list() {
    // pub because it is used by the box annihilator.
    let _ = try_take_task_borrow_list();
}

unsafe fn fail_borrowed(box: *mut raw::Box<()>, file: *c_char, line: size_t) {
    debug_borrow("fail_borrowed: ", box, 0, 0, file, line);

    match try_take_task_borrow_list() {
        None => { // not recording borrows
            let msg = "borrowed";
            do msg.as_c_str |msg_p| {
                sys::begin_unwind_(msg_p as *c_char, file, line);
            }
        }
        Some(borrow_list) => { // recording borrows
            let mut msg = ~"borrowed";
            let mut sep = " at ";
            foreach entry in borrow_list.rev_iter() {
                if entry.box == box {
                    msg.push_str(sep);
                    let filename = str::raw::from_c_str(entry.file);
                    msg.push_str(filename);
                    msg.push_str(fmt!(":%u", entry.line as uint));
                    sep = " and at ";
                }
            }
            do msg.as_c_str |msg_p| {
                sys::begin_unwind_(msg_p as *c_char, file, line)
            }
        }
    }
}

/// Because this code is so perf. sensitive, use a static constant so that
/// debug printouts are compiled out most of the time.
static ENABLE_DEBUG: bool = false;

#[inline]
unsafe fn debug_borrow<T>(tag: &'static str,
                          p: *const T,
                          old_bits: uint,
                          new_bits: uint,
                          filename: *c_char,
                          line: size_t) {
    //! A useful debugging function that prints a pointer + tag + newline
    //! without allocating memory.

    if ENABLE_DEBUG && ::rt::env::get().debug_borrow {
        debug_borrow_slow(tag, p, old_bits, new_bits, filename, line);
    }

    unsafe fn debug_borrow_slow<T>(tag: &'static str,
                                   p: *const T,
                                   old_bits: uint,
                                   new_bits: uint,
                                   filename: *c_char,
                                   line: size_t) {
        let dbg = STDERR_FILENO as io::fd_t;
        dbg.write_str(tag);
        dbg.write_hex(p as uint);
        dbg.write_str(" ");
        dbg.write_hex(old_bits);
        dbg.write_str(" ");
        dbg.write_hex(new_bits);
        dbg.write_str(" ");
        dbg.write_cstr(filename);
        dbg.write_str(":");
        dbg.write_hex(line as uint);
        dbg.write_str("\n");
    }
}

trait DebugPrints {
    fn write_hex(&self, val: uint);
    unsafe fn write_cstr(&self, str: *c_char);
}

impl DebugPrints for io::fd_t {
    fn write_hex(&self, mut i: uint) {
        let letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
                       '9', 'a', 'b', 'c', 'd', 'e', 'f'];
        static UINT_NIBBLES: uint = ::uint::bytes << 1;
        let mut buffer = [0_u8, ..UINT_NIBBLES+1];
        let mut c = UINT_NIBBLES;
        while c > 0 {
            c -= 1;
            buffer[c] = letters[i & 0xF] as u8;
            i >>= 4;
        }
        self.write(buffer.slice(0, UINT_NIBBLES));
    }

    unsafe fn write_cstr(&self, p: *c_char) {
        use libc::strlen;
        use vec;

        let len = strlen(p);
        let p: *u8 = transmute(p);
        do vec::raw::buf_as_slice(p, len as uint) |s| {
            self.write(s);
        }
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
                let err = fmt!("wrong borrow found, br=%?", br);
                do err.as_c_str |msg_p| {
                    sys::begin_unwind_(msg_p as *c_char, file, line)
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


extern {
    #[rust_stack]
    pub fn rust_take_task_borrow_list(task: *rust_task) -> *c_void;

    #[rust_stack]
    pub fn rust_set_task_borrow_list(task: *rust_task, map: *c_void);

    #[rust_stack]
    pub fn rust_try_get_task() -> *rust_task;
}
