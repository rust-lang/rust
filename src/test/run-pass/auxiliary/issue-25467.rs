// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="lib"]

pub trait Trait {
    // the issue is sensitive to interning order - so use names
    // unlikely to appear in libstd.
    type Issue25467FooT;
    type Issue25467BarT;
}

pub type Object = Option<Box<Trait<Issue25467FooT=(),Issue25467BarT=()>>>;
