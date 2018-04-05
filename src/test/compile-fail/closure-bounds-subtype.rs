// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn take_any<F>(_: F) where F: FnOnce() {
}

fn take_const_owned<F>(_: F) where F: FnOnce() + Sync + Send {
}

fn give_any<F>(f: F) where F: FnOnce() {
    take_any(f);
}

fn give_owned<F>(f: F) where F: FnOnce() + Send {
    take_any(f);
    take_const_owned(f); //~ ERROR `F` cannot be shared between threads safely [E0277]
}

fn main() {}
