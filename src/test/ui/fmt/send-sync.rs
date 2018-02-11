// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn send<T: Send>(_: T) {}
fn sync<T: Sync>(_: T) {}

fn main() {
    // `Cell` is not `Sync`, so `&Cell` is neither `Sync` nor `Send`,
    // `std::fmt::Arguments` used to forget this...
    let c = std::cell::Cell::new(42);
    send(format_args!("{:?}", c)); //~ ERROR E0277
    sync(format_args!("{:?}", c)); //~ ERROR E0277
}
