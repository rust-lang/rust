// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::raw::*;
use core::mem;

#[test]
fn synthesize_closure() {
    unsafe {
        let x = 10;
        let f: |int| -> int = |y| x + y;

        assert_eq!(f(20), 30);

        let original_closure: Closure = mem::transmute(f);

        let actual_function_pointer = original_closure.code;
        let environment = original_closure.env;

        let new_closure = Closure {
            code: actual_function_pointer,
            env: environment
        };

        let new_f: |int| -> int = mem::transmute(new_closure);
        assert_eq!(new_f(20), 30);
    }
}
