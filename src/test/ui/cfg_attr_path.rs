// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![deny(unused_attributes)] // c.f #35584

mod auxiliary {
    #[cfg_attr(any(), path = "nonexistent_file.rs")] pub mod namespaced_enums;
    #[cfg_attr(all(), path = "namespaced_enums.rs")] pub mod nonexistent_file;
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let _ = auxiliary::namespaced_enums::Foo::A;
    let _ = auxiliary::nonexistent_file::Foo::A;
}
