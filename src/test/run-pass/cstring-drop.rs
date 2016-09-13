// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten

// Test that `CString::new("hello").unwrap().as_ptr()` pattern
// leads to failure.

use std::env;
use std::ffi::{CString, CStr};
use std::os::raw::c_char;
use std::process::{Command, Stdio};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "child" {
        // Repeat several times to be more confident that
        // it is `Drop` for `CString` that does the cleanup,
        // and not just some lucky UB.
        let xs = vec![CString::new("Hello").unwrap(); 10];
        let ys = xs.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        drop(xs);
        assert!(ys.into_iter().any(is_hello));
        return;
    }

    let output = Command::new(&args[0]).arg("child").output().unwrap();
    assert!(!output.status.success());
}

fn is_hello(s: *const c_char) -> bool {
    // `s` is a dangling pointer and reading it is technically
    // undefined behavior. But we want to prevent the most diabolical
    // kind of UB (apart from nasal demons): reading a value that was
    // previously written.
    //
    // Segfaulting or reading an empty string is Ok,
    // reading "Hello" is bad.
    let s = unsafe { CStr::from_ptr(s) };
    let hello = CString::new("Hello").unwrap();
    s == hello.as_ref()
}
