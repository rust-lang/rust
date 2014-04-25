// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::dynamic_lib::DynamicLibrary;
use std::os;

pub fn main() {
    unsafe {
        let path = Path::new("libdylib.so");
        let a = DynamicLibrary::open(Some(&path)).unwrap();
        assert!(a.symbol::<int>("fun1").is_ok());
        assert!(a.symbol::<int>("fun2").is_err());
        assert!(a.symbol::<int>("fun3").is_err());
        assert!(a.symbol::<int>("fun4").is_ok());
        assert!(a.symbol::<int>("fun5").is_ok());
    }
}
