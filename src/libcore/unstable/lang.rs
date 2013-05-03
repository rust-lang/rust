// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Runtime calls emitted by the compiler.

use uint;
use cast::transmute;
use libc::{c_char, c_uchar, c_void, size_t, uintptr_t, c_int, STDERR_FILENO};
use managed::raw::BoxRepr;
use str;
use sys;
use unstable::exchange_alloc;
use cast::transmute;
use task::rt::rust_get_task;
use option::{Option, Some, None};

#[allow(non_camel_case_types)]
pub type rust_task = c_void;

pub static FROZEN_BIT: uint = 1 << (uint::bits - 1);
pub static MUT_BIT: uint = 1 << (uint::bits - 2);
static ALL_BITS: uint = FROZEN_BIT | MUT_BIT;

pub mod rustrt {
    use unstable::lang::rust_task;
    use libc::{c_void, c_char, uintptr_t};

    pub extern {
        #[rust_stack]
        unsafe fn rust_upcall_malloc(td: *c_char, size: uintptr_t) -> *c_char;

        #[rust_stack]
        unsafe fn rust_upcall_free(ptr: *c_char);

        #[fast_ffi]
        unsafe fn rust_upcall_malloc_noswitch(td: *c_char,
                                              size: uintptr_t)
                                           -> *c_char;

        #[fast_ffi]
        unsafe fn rust_upcall_free_noswitch(ptr: *c_char);

        #[rust_stack]
        fn rust_take_task_borrow_list(task: *rust_task) -> *c_void;

        #[rust_stack]
        fn rust_set_task_borrow_list(task: *rust_task, map: *c_void);

        fn rust_dbg_breakpoint();
    }
}

#[lang="fail_"]
pub fn fail_(expr: *c_char, file: *c_char, line: size_t) -> ! {
    sys::begin_unwind_(expr, file, line);
}

#[lang="fail_bounds_check"]
pub fn fail_bounds_check(file: *c_char, line: size_t,
                         index: size_t, len: size_t) {
    let msg = fmt!("index out of bounds: the len is %d but the index is %d",
                    len as int, index as int);
    do str::as_buf(msg) |p, _len| {
        fail_(p as *c_char, file, line);
    }
}

#[deriving(Eq)]
struct BorrowRecord {
    box: *mut BoxRepr,
    file: *c_char,
    line: size_t
}

fn try_take_task_borrow_list() -> Option<~[BorrowRecord]> {
    unsafe {
        let cur_task = rust_get_task();
        let ptr = rustrt::rust_take_task_borrow_list(cur_task);
        if ptr.is_null() {
            None
        } else {
            let v: ~[BorrowRecord] = transmute(ptr);
            Some(v)
        }
    }
}

fn swap_task_borrow_list(f: &fn(~[BorrowRecord]) -> ~[BorrowRecord]) {
    unsafe {
        let cur_task = rust_get_task();
        let mut borrow_list: ~[BorrowRecord] = {
            let ptr = rustrt::rust_take_task_borrow_list(cur_task);
            if ptr.is_null() { ~[] } else { transmute(ptr) }
        };
        borrow_list = f(borrow_list);
        rustrt::rust_set_task_borrow_list(cur_task, transmute(borrow_list));
    }
}

pub unsafe fn clear_task_borrow_list() {
    // pub because it is used by the box annihilator.
    let _ = try_take_task_borrow_list();
}

fn fail_borrowed(box: *mut BoxRepr, file: *c_char, line: size_t) {
    debug_ptr("fail_borrowed: ", box);

    match try_take_task_borrow_list() {
        None => { // not recording borrows
            let msg = "borrowed";
            do str::as_buf(msg) |msg_p, _| {
                fail_(msg_p as *c_char, file, line);
            }
        }
        Some(borrow_list) => { // recording borrows
            let mut msg = ~"borrowed";
            let mut sep = " at ";
            for borrow_list.each_reverse |entry| {
                if entry.box == box {
                    str::push_str(&mut msg, sep);
                    let filename = unsafe {
                        str::raw::from_c_str(entry.file)
                    };
                    str::push_str(&mut msg, filename);
                    str::push_str(&mut msg, fmt!(":%u", entry.line as uint));
                    sep = " and at ";
                }
            }
            do str::as_buf(msg) |msg_p, _| {
                fail_(msg_p as *c_char, file, line)
            }
        }
    }
}

// FIXME #4942: Make these signatures agree with exchange_alloc's signatures
#[lang="exchange_malloc"]
#[inline(always)]
pub unsafe fn exchange_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    let result = transmute(exchange_alloc::malloc(transmute(td), transmute(size)));
    debug_ptr("exchange_malloc: ", result);
    return result;
}

/// Because this code is so perf. sensitive, use a static constant so that
/// debug printouts are compiled out most of the time.
static ENABLE_DEBUG_PTR: bool = true;

#[inline]
pub fn debug_ptr<T>(tag: &'static str, p: *const T) {
    //! A useful debugging function that prints a pointer + tag + newline
    //! without allocating memory.

    if ENABLE_DEBUG_PTR && ::rt::env::get().debug_mem {
        debug_ptr_slow(tag, p);
    }

    fn debug_ptr_slow<T>(tag: &'static str, p: *const T) {
        use io;
        let dbg = STDERR_FILENO as io::fd_t;
        let letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
                       '9', 'a', 'b', 'c', 'd', 'e', 'f'];
        dbg.write_str(tag);

        static uint_nibbles: uint = ::uint::bytes << 1;
        let mut buffer = [0_u8, ..uint_nibbles+1];
        let mut i = p as uint;
        let mut c = uint_nibbles;
        while c > 0 {
            c -= 1;
            buffer[c] = letters[i & 0xF] as u8;
            i >>= 4;
        }
        dbg.write(buffer.slice(0, uint_nibbles));

        dbg.write_str("\n");
    }
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[lang="exchange_free"]
#[inline(always)]
pub unsafe fn exchange_free(ptr: *c_char) {
    debug_ptr("exchange_free: ", ptr);
    exchange_alloc::free(transmute(ptr))
}

#[lang="malloc"]
#[inline(always)]
pub unsafe fn local_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    let result = rustrt::rust_upcall_malloc_noswitch(td, size);
    debug_ptr("local_malloc: ", result);
    return result;
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[lang="free"]
#[inline(always)]
pub unsafe fn local_free(ptr: *c_char) {
    debug_ptr("local_free: ", ptr);
    rustrt::rust_upcall_free_noswitch(ptr);
}

#[cfg(stage0)]
#[lang="borrow_as_imm"]
#[inline(always)]
pub unsafe fn borrow_as_imm(a: *u8) {
    let a: *mut BoxRepr = transmute(a);
    (*a).header.ref_count |= FROZEN_BIT;
}

#[cfg(not(stage0))]
#[lang="borrow_as_imm"]
#[inline(always)]
pub unsafe fn borrow_as_imm(a: *u8, file: *c_char, line: size_t) -> uint {
    let a: *mut BoxRepr = transmute(a);
    let ref_count = (*a).header.ref_count;

    debug_ptr("borrow_as_imm (ptr) :", a);
    debug_ptr("              (ref) :", ref_count as *());
    debug_ptr("              (line): ", line as *());

    if (ref_count & MUT_BIT) != 0 {
        fail_borrowed(a, file, line);
    }

    ref_count
}

#[cfg(not(stage0))]
#[lang="borrow_as_mut"]
#[inline(always)]
pub unsafe fn borrow_as_mut(a: *u8, file: *c_char, line: size_t) -> uint {
    let a: *mut BoxRepr = transmute(a);

    debug_ptr("borrow_as_mut (ptr): ", a);
    debug_ptr("              (line): ", line as *());

    let ref_count = (*a).header.ref_count;
    if (ref_count & (MUT_BIT|FROZEN_BIT)) != 0 {
        fail_borrowed(a, file, line);
    }
    ref_count
}


#[cfg(not(stage0))]
#[lang="record_borrow"]
pub unsafe fn record_borrow(a: *u8, old_ref_count: uint,
                            file: *c_char, line: size_t) {
    if (old_ref_count & ALL_BITS) == 0 {
        // was not borrowed before
        let a: *mut BoxRepr = transmute(a);
        do swap_task_borrow_list |borrow_list| {
            let mut borrow_list = borrow_list;
            borrow_list.push(BorrowRecord {box: a, file: file, line: line});
            borrow_list
        }
    }
}

#[cfg(not(stage0))]
#[lang="unrecord_borrow"]
pub unsafe fn unrecord_borrow(a: *u8, old_ref_count: uint,
                              file: *c_char, line: size_t) {
    if (old_ref_count & ALL_BITS) == 0 {
        // was not borrowed before
        let a: *mut BoxRepr = transmute(a);
        do swap_task_borrow_list |borrow_list| {
            let mut borrow_list = borrow_list;
            let br = BorrowRecord {box: a, file: file, line: line};
            match borrow_list.rposition_elem(&br) {
                Some(idx) => {
                    borrow_list.remove(idx);
                    borrow_list
                }
                None => {
                    let err = fmt!("no borrow found, br=%?, borrow_list=%?",
                                   br, borrow_list);
                    do str::as_buf(err) |msg_p, _| {
                        fail_(msg_p as *c_char, file, line)
                    }
                }
            }
        }
    }
}

#[cfg(stage0)]
#[lang="return_to_mut"]
#[inline(always)]
pub unsafe fn return_to_mut(a: *u8) {
    // Sometimes the box is null, if it is conditionally frozen.
    // See e.g. #4904.
    if !a.is_null() {
        let a: *mut BoxRepr = transmute(a);
        (*a).header.ref_count &= !FROZEN_BIT;
    }
}

#[cfg(not(stage0))]
#[lang="return_to_mut"]
#[inline(always)]
pub unsafe fn return_to_mut(a: *u8, old_ref_count: uint,
                            file: *c_char, line: size_t) {
    // Sometimes the box is null, if it is conditionally frozen.
    // See e.g. #4904.
    if !a.is_null() {
        let a: *mut BoxRepr = transmute(a);
        let ref_count = (*a).header.ref_count;
        let combined = (ref_count & !ALL_BITS) | (old_ref_count & ALL_BITS);
        (*a).header.ref_count = combined;

        debug_ptr("return_to_mut (ptr) : ", a);
        debug_ptr("              (line): ", line as *());
        debug_ptr("              (old) : ", old_ref_count as *());
        debug_ptr("              (new) : ", ref_count as *());
        debug_ptr("              (comb): ", combined as *());
    }
}

#[cfg(stage0)]
#[lang="check_not_borrowed"]
#[inline(always)]
pub unsafe fn check_not_borrowed(a: *u8) {
    let a: *mut BoxRepr = transmute(a);
    if ((*a).header.ref_count & FROZEN_BIT) != 0 {
        do str::as_buf("XXX") |file_p, _| {
            fail_borrowed(a, file_p as *c_char, 0);
        }
    }
}

#[cfg(not(stage0))]
#[lang="check_not_borrowed"]
#[inline(always)]
pub unsafe fn check_not_borrowed(a: *u8,
                                 file: *c_char,
                                 line: size_t) {
    let a: *mut BoxRepr = transmute(a);
    if ((*a).header.ref_count & FROZEN_BIT) != 0 {
        fail_borrowed(a, file, line);
    }
}

#[lang="strdup_uniq"]
#[inline(always)]
pub unsafe fn strdup_uniq(ptr: *c_uchar, len: uint) -> ~str {
    str::raw::from_buf_len(ptr, len)
}

#[lang="start"]
pub fn start(main: *u8, argc: int, argv: **c_char,
             crate_map: *u8) -> int {
    use libc::getenv;
    use rt::start;

    unsafe {
        let use_old_rt = do str::as_c_str("RUST_NEWRT") |s| {
            getenv(s).is_null()
        };
        if use_old_rt {
            return rust_start(main as *c_void, argc as c_int, argv,
                              crate_map as *c_void) as int;
        } else {
            return start(main, argc, argv, crate_map);
        }
    }

    extern {
        fn rust_start(main: *c_void, argc: c_int, argv: **c_char,
                      crate_map: *c_void) -> c_int;
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
