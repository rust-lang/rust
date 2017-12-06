// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use mem;
use sys::c;

pub fn hashmap_random_keys() -> (u64, u64) {
    let mut v = (0, 0);
    let ret = unsafe {
        c::RtlGenRandom(&mut v as *mut _ as *mut u8,
                        mem::size_of_val(&v) as c::ULONG)
    };
    if ret == 0 {
        panic!("couldn't generate random bytes: {}",
               io::Error::last_os_error());
    }
    return v
}
