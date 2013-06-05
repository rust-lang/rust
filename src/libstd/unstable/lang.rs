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
use rt::{context, OldTaskContext};
use rt::task::Task;
use rt::local::Local;
use option::{Option, Some, None};
use io;
use rt::global_heap;

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

        #[rust_stack]
        fn rust_try_get_task() -> *rust_task;

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
        let cur_task: *rust_task = rustrt::rust_try_get_task();
        if cur_task.is_not_null() {
            let ptr = rustrt::rust_take_task_borrow_list(cur_task);
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
        let cur_task: *rust_task = rustrt::rust_try_get_task();
        if cur_task.is_not_null() {
            let mut borrow_list: ~[BorrowRecord] = {
                let ptr = rustrt::rust_take_task_borrow_list(cur_task);
                if ptr.is_null() { ~[] } else { transmute(ptr) }
            };
            borrow_list = f(borrow_list);
            rustrt::rust_set_task_borrow_list(cur_task, transmute(borrow_list));
        }
    }
}

pub unsafe fn clear_task_borrow_list() {
    // pub because it is used by the box annihilator.
    let _ = try_take_task_borrow_list();
}

unsafe fn fail_borrowed(box: *mut BoxRepr, file: *c_char, line: size_t) {
    debug_borrow("fail_borrowed: ", box, 0, 0, file, line);

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
                    let filename = str::raw::from_c_str(entry.file);
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
    transmute(global_heap::malloc(transmute(td), transmute(size)))
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
        static uint_nibbles: uint = ::uint::bytes << 1;
        let mut buffer = [0_u8, ..uint_nibbles+1];
        let mut c = uint_nibbles;
        while c > 0 {
            c -= 1;
            buffer[c] = letters[i & 0xF] as u8;
            i >>= 4;
        }
        self.write(buffer.slice(0, uint_nibbles));
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

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[lang="exchange_free"]
#[inline(always)]
pub unsafe fn exchange_free(ptr: *c_char) {
    global_heap::free(transmute(ptr))
}

#[lang="malloc"]
pub unsafe fn local_malloc(td: *c_char, size: uintptr_t) -> *c_char {
    match context() {
        OldTaskContext => {
            return rustrt::rust_upcall_malloc_noswitch(td, size);
        }
        _ => {
            let mut alloc = ::ptr::null();
            do Local::borrow::<Task> |task| {
                alloc = task.heap.alloc(td as *c_void, size as uint) as *c_char;
            }
            return alloc;
        }
    }
}

// NB: Calls to free CANNOT be allowed to fail, as throwing an exception from
// inside a landing pad may corrupt the state of the exception handler. If a
// problem occurs, call exit instead.
#[lang="free"]
pub unsafe fn local_free(ptr: *c_char) {
    match context() {
        OldTaskContext => {
            rustrt::rust_upcall_free_noswitch(ptr);
        }
        _ => {
            do Local::borrow::<Task> |task| {
                task.heap.free(ptr as *c_void);
            }
        }
    }
}

#[lang="borrow_as_imm"]
#[inline(always)]
pub unsafe fn borrow_as_imm(a: *u8, file: *c_char, line: size_t) -> uint {
    let a: *mut BoxRepr = transmute(a);
    let old_ref_count = (*a).header.ref_count;
    let new_ref_count = old_ref_count | FROZEN_BIT;

    debug_borrow("borrow_as_imm:", a, old_ref_count, new_ref_count, file, line);

    if (old_ref_count & MUT_BIT) != 0 {
        fail_borrowed(a, file, line);
    }

    (*a).header.ref_count = new_ref_count;

    old_ref_count
}

#[lang="borrow_as_mut"]
#[inline(always)]
pub unsafe fn borrow_as_mut(a: *u8, file: *c_char, line: size_t) -> uint {
    let a: *mut BoxRepr = transmute(a);
    let old_ref_count = (*a).header.ref_count;
    let new_ref_count = old_ref_count | MUT_BIT | FROZEN_BIT;

    debug_borrow("borrow_as_mut:", a, old_ref_count, new_ref_count, file, line);

    if (old_ref_count & (MUT_BIT|FROZEN_BIT)) != 0 {
        fail_borrowed(a, file, line);
    }

    (*a).header.ref_count = new_ref_count;

    old_ref_count
}


#[lang="record_borrow"]
pub unsafe fn record_borrow(a: *u8, old_ref_count: uint,
                            file: *c_char, line: size_t) {
    if (old_ref_count & ALL_BITS) == 0 {
        // was not borrowed before
        let a: *mut BoxRepr = transmute(a);
        debug_borrow("record_borrow:", a, old_ref_count, 0, file, line);
        do swap_task_borrow_list |borrow_list| {
            let mut borrow_list = borrow_list;
            borrow_list.push(BorrowRecord {box: a, file: file, line: line});
            borrow_list
        }
    }
}

#[lang="unrecord_borrow"]
pub unsafe fn unrecord_borrow(a: *u8, old_ref_count: uint,
                              file: *c_char, line: size_t) {
    if (old_ref_count & ALL_BITS) == 0 {
        // was not borrowed before, so we should find the record at
        // the end of the list
        let a: *mut BoxRepr = transmute(a);
        debug_borrow("unrecord_borrow:", a, old_ref_count, 0, file, line);
        do swap_task_borrow_list |borrow_list| {
            let mut borrow_list = borrow_list;
            assert!(!borrow_list.is_empty());
            let br = borrow_list.pop();
            if br.box != a || br.file != file || br.line != line {
                let err = fmt!("wrong borrow found, br=%?", br);
                do str::as_buf(err) |msg_p, _| {
                    fail_(msg_p as *c_char, file, line)
                }
            }
            borrow_list
        }
    }
}

#[lang="return_to_mut"]
#[inline(always)]
pub unsafe fn return_to_mut(a: *u8, orig_ref_count: uint,
                            file: *c_char, line: size_t) {
    // Sometimes the box is null, if it is conditionally frozen.
    // See e.g. #4904.
    if !a.is_null() {
        let a: *mut BoxRepr = transmute(a);
        let old_ref_count = (*a).header.ref_count;
        let new_ref_count =
            (old_ref_count & !ALL_BITS) | (orig_ref_count & ALL_BITS);

        debug_borrow("return_to_mut:",
                     a, old_ref_count, new_ref_count, file, line);

        (*a).header.ref_count = new_ref_count;
    }
}

#[lang="check_not_borrowed"]
#[inline(always)]
pub unsafe fn check_not_borrowed(a: *u8,
                                 file: *c_char,
                                 line: size_t) {
    let a: *mut BoxRepr = transmute(a);
    let ref_count = (*a).header.ref_count;
    debug_borrow("check_not_borrowed:", a, ref_count, 0, file, line);
    if (ref_count & FROZEN_BIT) != 0 {
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
    use rt;
    use sys::Closure;
    use ptr;
    use cast;
    use os;

    unsafe {
        let use_old_rt = os::getenv("RUST_NEWRT").is_none();
        if use_old_rt {
            return rust_start(main as *c_void, argc as c_int, argv,
                              crate_map as *c_void) as int;
        } else {
            return do rt::start(argc, argv as **u8, crate_map) {
                unsafe {
                    // `main` is an `fn() -> ()` that doesn't take an environment
                    // XXX: Could also call this as an `extern "Rust" fn` once they work
                    let main = Closure {
                        code: main as *(),
                        env: ptr::null(),
                    };
                    let mainfn: &fn() = cast::transmute(main);

                    mainfn();
                }
            };
        }
    }

    extern {
        fn rust_start(main: *c_void, argc: c_int, argv: **c_char,
                      crate_map: *c_void) -> c_int;
    }
}
