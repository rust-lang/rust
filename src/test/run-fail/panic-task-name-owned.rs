// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:thread 'owned name' panicked at 'test'
// ignore-emscripten Needs threads.

use std::thread::Builder;

fn main() {
    let r: () = Builder::new()
                    .name("owned name".to_string())
                    .spawn(move || {
                        panic!("test");
                        ()
                    })
                    .unwrap()
                    .join()
                    .unwrap();
    panic!();
}
