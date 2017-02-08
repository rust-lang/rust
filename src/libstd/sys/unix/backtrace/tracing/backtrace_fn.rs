// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// As always - iOS on arm uses SjLj exceptions and
/// _Unwind_Backtrace is even not available there. Still,
/// backtraces could be extracted using a backtrace function,
/// which thanks god is public
///
/// As mentioned in a huge comment block in `super::super`, backtrace
/// doesn't play well with green threads, so while it is extremely nice and
/// simple to use it should be used only on iOS devices as the only viable
/// option.

use io::prelude::*;
use io;
use libc;
use mem;
use sys::backtrace::BacktraceContext;
use sys_common::backtrace::Frame;

#[inline(never)] // if we know this is a function call, we can skip it when
                 // tracing
pub fn unwind_backtrace(frames: &mut [Frame])
    -> io::Result<(usize, BacktraceContext)>
{
    const FRAME_LEN = 100;
    assert!(FRAME_LEN >= frames);
    let mut raw_frames = [0; FRAME_LEN];
    let nb_frames = backtrace(frames.as_mut_ptr(),
                              frames.len() as libc::c_int);
    for (from, to) in raw_frames[..nb_frames].iter().zip(frames.iter_mut()) {
        *to = Frame {
            exact_position: *from,
            symbol_addr: *from,
        };
    }
    (nb_frames as usize, BacktraceContext)
}

extern fn backtrace(buf: *mut *mut libc::c_void, sz: libc::c_int) -> libc::c_int;
