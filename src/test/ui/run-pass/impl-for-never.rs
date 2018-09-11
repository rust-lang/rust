// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can call static methods on ! both directly and when it appears in a generic

#![feature(never_type)]

trait StringifyType {
    fn stringify_type() -> &'static str;
}

impl StringifyType for ! {
    fn stringify_type() -> &'static str {
        "!"
    }
}

fn maybe_stringify<T: StringifyType>(opt: Option<T>) -> &'static str {
    match opt {
        Some(_) => T::stringify_type(),
        None => "none",
    }
}

fn main() {
    println!("! is {}", <!>::stringify_type());
    println!("None is {}", maybe_stringify(None::<!>));
}

