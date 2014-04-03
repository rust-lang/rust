// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// ignore-win32 dynamic_lib can read dllexported symbols only
// ignore-linux apparently dlsym doesn't work on program symbols?
// ignore-android apparently dlsym doesn't work on program symbols?
// ignore-freebsd apparently dlsym doesn't work on program symbols?

use std::unstable::dynamic_lib::DynamicLibrary;

#[no_mangle] pub extern "C" fn fun1() {}
#[no_mangle] extern "C" fn fun2() {}

mod foo {
    #[no_mangle] pub extern "C" fn fun3() {}
}
pub mod bar {
    #[no_mangle] pub extern "C" fn fun4() {}
}

#[no_mangle] pub fn fun5() {}

pub fn main() {
    unsafe {
        let a = DynamicLibrary::open(None).unwrap();
        assert!(a.symbol::<int>("fun1").is_ok());
        assert!(a.symbol::<int>("fun2").is_err());
        assert!(a.symbol::<int>("fun3").is_err());
        assert!(a.symbol::<int>("fun4").is_ok());
        assert!(a.symbol::<int>("fun5").is_err());
    }
}
