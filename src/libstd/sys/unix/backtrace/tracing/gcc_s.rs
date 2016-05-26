// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use io::prelude::*;
use libc;
use mem;
use sys_common::mutex::Mutex;

use super::super::printing::print;
use unwind as uw;

#[inline(never)] // if we know this is a function call, we can skip it when
                 // tracing
pub fn write(w: &mut Write) -> io::Result<()> {
    struct Context<'a> {
        idx: isize,
        writer: &'a mut (Write+'a),
        last_error: Option<io::Error>,
    }

    // When using libbacktrace, we use some necessary global state, so we
    // need to prevent more than one thread from entering this block. This
    // is semi-reasonable in terms of printing anyway, and we know that all
    // I/O done here is blocking I/O, not green I/O, so we don't have to
    // worry about this being a native vs green mutex.
    static LOCK: Mutex = Mutex::new();
    unsafe {
        LOCK.lock();

        writeln!(w, "stack backtrace:")?;

        let mut cx = Context { writer: w, last_error: None, idx: 0 };
        let ret = match {
            uw::_Unwind_Backtrace(trace_fn,
                                  &mut cx as *mut Context as *mut libc::c_void)
        } {
            uw::_URC_NO_REASON => {
                match cx.last_error {
                    Some(err) => Err(err),
                    None => Ok(())
                }
            }
            _ => Ok(()),
        };
        LOCK.unlock();
        return ret
    }

    extern fn trace_fn(ctx: *mut uw::_Unwind_Context,
                       arg: *mut libc::c_void) -> uw::_Unwind_Reason_Code {
        let cx: &mut Context = unsafe { mem::transmute(arg) };
        let mut ip_before_insn = 0;
        let mut ip = unsafe {
            uw::_Unwind_GetIPInfo(ctx, &mut ip_before_insn) as *mut libc::c_void
        };
        if !ip.is_null() && ip_before_insn == 0 {
            // this is a non-signaling frame, so `ip` refers to the address
            // after the calling instruction. account for that.
            ip = (ip as usize - 1) as *mut _;
        }

        // dladdr() on osx gets whiny when we use FindEnclosingFunction, and
        // it appears to work fine without it, so we only use
        // FindEnclosingFunction on non-osx platforms. In doing so, we get a
        // slightly more accurate stack trace in the process.
        //
        // This is often because panic involves the last instruction of a
        // function being "call std::rt::begin_unwind", with no ret
        // instructions after it. This means that the return instruction
        // pointer points *outside* of the calling function, and by
        // unwinding it we go back to the original function.
        let symaddr = if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
            ip
        } else {
            unsafe { uw::_Unwind_FindEnclosingFunction(ip) }
        };

        // Don't print out the first few frames (they're not user frames)
        cx.idx += 1;
        if cx.idx <= 0 { return uw::_URC_NO_REASON }
        // Don't print ginormous backtraces
        if cx.idx > 100 {
            match write!(cx.writer, " ... <frames omitted>\n") {
                Ok(()) => {}
                Err(e) => { cx.last_error = Some(e); }
            }
            return uw::_URC_FAILURE
        }

        // Once we hit an error, stop trying to print more frames
        if cx.last_error.is_some() { return uw::_URC_FAILURE }

        match print(cx.writer, cx.idx, ip, symaddr) {
            Ok(()) => {}
            Err(e) => { cx.last_error = Some(e); }
        }

        // keep going
        uw::_URC_NO_REASON
    }
}
