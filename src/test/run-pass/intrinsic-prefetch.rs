// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod rusti {
    extern "rust-intrinsic" {
        pub fn prefetch<T>(address: *T);
        pub fn prefetch_write<T>(address: *mut T);
    }
}

fn main() {
    let i = 1;
    let mut j = 2;
    unsafe {
        let ip: *int = &i;
        let jp: *mut int = &mut j;
        let np: *int = 0 as *int;

        rusti::prefetch(ip);
        rusti::prefetch_write(jp);

        // prefetches can't trap
        rusti::prefetch(np);
    }
}
