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
use sys::mutex::Mutex;

use super::super::printing::print;

#[inline(never)]
pub fn write(w: &mut Write) -> io::Result<()> {
    extern {
        fn backtrace(buf: *mut *mut libc::c_void,
                     sz: libc::c_int) -> libc::c_int;
    }

    // while it doesn't requires lock for work as everything is
    // local, it still displays much nicer backtraces when a
    // couple of threads panic simultaneously
    static LOCK: Mutex = Mutex::new();
    unsafe {
        LOCK.lock();

        writeln!(w, "stack backtrace:")?;
        // 100 lines should be enough
        const SIZE: usize = 100;
        let mut buf: [*mut libc::c_void; SIZE] = mem::zeroed();
        let cnt = backtrace(buf.as_mut_ptr(), SIZE as libc::c_int) as usize;

        // skipping the first one as it is write itself
        for i in 1..cnt {
            print(w, i as isize, buf[i], buf[i])?
        }
        LOCK.unlock();
    }
    Ok(())
}
