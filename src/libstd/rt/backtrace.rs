// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple backtrace functionality (to print on panic)

#![allow(non_camel_case_types)]

use option::Option::{Some, None};
use os;
use sync::atomic;

pub use sys::backtrace::write;

// For now logging is turned off by default, and this function checks to see
// whether the magical environment variable is present to see if it's turned on.
pub fn log_enabled() -> bool {
    static ENABLED: atomic::AtomicInt = atomic::INIT_ATOMIC_INT;
    match ENABLED.load(atomic::SeqCst) {
        1 => return false,
        2 => return true,
        _ => {}
    }

    let val = match os::getenv("RUST_BACKTRACE") {
        Some(..) => 2,
        None => 1,
    };
    ENABLED.store(val, atomic::SeqCst);
    val == 2
}

#[cfg(test)]
mod test {
    use prelude::*;
    use sys_common;
    macro_rules! t { ($a:expr, $b:expr) => ({
        let mut m = Vec::new();
        sys_common::backtrace::demangle(&mut m, $a).unwrap();
        assert_eq!(String::from_utf8(m).unwrap(), $b);
    }) }

    #[test]
    fn demangle() {
        t!("test", "test");
        t!("_ZN4testE", "test");
        t!("_ZN4test", "_ZN4test");
        t!("_ZN4test1a2bcE", "test::a::bc");
    }

    #[test]
    fn demangle_dollars() {
        t!("_ZN4$UP$E", "Box");
        t!("_ZN8$UP$testE", "Boxtest");
        t!("_ZN8$UP$test4foobE", "Boxtest::foob");
        t!("_ZN8$x20test4foobE", " test::foob");
    }

    #[test]
    fn demangle_many_dollars() {
        t!("_ZN12test$x20test4foobE", "test test::foob");
        t!("_ZN12test$UP$test4foobE", "testBoxtest::foob");
    }

    #[test]
    fn demangle_windows() {
        t!("ZN4testE", "test");
        t!("ZN12test$x20test4foobE", "test test::foob");
        t!("ZN12test$UP$test4foobE", "testBoxtest::foob");
    }
}
