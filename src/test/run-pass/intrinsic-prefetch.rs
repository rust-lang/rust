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
    let j = box 1;
    unsafe {
        let ip: *int = &i;
        let jp: *mut int = std::mem::transmute(j);

        rusti::prefetch(ip);
        rusti::prefetch_write(jp);

        drop(std::mem::transmute::<_, Box<int>>(jp));
    }
}
