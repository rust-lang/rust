// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

#[derive(Copy, Clone)]
pub struct Quad { a: u64, b: u64, c: u64, d: u64 }

mod rustrt {
    use super::Quad;

    #[link(name = "rust_test_helpers")]
    extern {
        pub fn get_c_many_params(_: *const (), _: *const (),
                                 _: *const (), _: *const (), f: Quad) -> u64;
    }
}

fn test() {
    unsafe {
        let null = std::ptr::null();
        let q = Quad {
            a: 1,
            b: 2,
            c: 3,
            d: 4
        };
        assert_eq!(rustrt::get_c_many_params(null, null, null, null, q), q.c);
    }
}

pub fn main() {
    test();
}
