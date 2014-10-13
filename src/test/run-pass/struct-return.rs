// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15883

pub struct Quad { a: u64, b: u64, c: u64, d: u64 }
pub struct Floats { a: f64, b: u8, c: f64 }

mod rustrt {
    use super::{Floats, Quad};

    #[link(name = "rust_test_helpers")]
    extern {
        pub fn rust_dbg_abi_1(q: Quad) -> Quad;
        pub fn rust_dbg_abi_2(f: Floats) -> Floats;
    }
}

fn test1() {
    unsafe {
        let q = Quad { a: 0xaaaa_aaaa_aaaa_aaaa_u64,
                 b: 0xbbbb_bbbb_bbbb_bbbb_u64,
                 c: 0xcccc_cccc_cccc_cccc_u64,
                 d: 0xdddd_dddd_dddd_dddd_u64 };
        let qq = rustrt::rust_dbg_abi_1(q);
        println!("a: {:x}", qq.a as uint);
        println!("b: {:x}", qq.b as uint);
        println!("c: {:x}", qq.c as uint);
        println!("d: {:x}", qq.d as uint);
        assert_eq!(qq.a, q.c + 1u64);
        assert_eq!(qq.b, q.d - 1u64);
        assert_eq!(qq.c, q.a + 1u64);
        assert_eq!(qq.d, q.b - 1u64);
    }
}

#[cfg(target_arch = "x86_64")]
fn test2() {
    unsafe {
        let f = Floats { a: 1.234567890e-15_f64,
                 b: 0b_1010_1010_u8,
                 c: 1.0987654321e-15_f64 };
        let ff = rustrt::rust_dbg_abi_2(f);
        println!("a: {}", ff.a as f64);
        println!("b: {}", ff.b as uint);
        println!("c: {}", ff.c as f64);
        assert_eq!(ff.a, f.c + 1.0f64);
        assert_eq!(ff.b, 0xff_u8);
        assert_eq!(ff.c, f.a - 1.0f64);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "arm"))]
fn test2() {
}

pub fn main() {
    test1();
    test2();
}
