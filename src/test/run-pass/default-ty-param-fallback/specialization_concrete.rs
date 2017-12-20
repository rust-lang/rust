// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(default_type_parameter_fallback)]
#![feature(specialization)]

use std::fmt::Debug;

trait B {
    fn b(&self) -> Self;
}

default impl<T=i32> B for Option<T>
    where T: Default
{
    fn b(&self) -> Option<T> {
        Some(T::default())
    }
}

impl B for Option<String>
{
    fn b(&self) -> Option<String> {
        Some("special".into())
    }
}

fn main() {
    let x = None;
    // Even with defaults, the concrete impl is prefered,
    // the defaults are simply ignored.
    foo(x.b());
}

fn foo<T=i32>(a: Option<T>)
    where T: Debug {
    assert_eq!(format!("{:?}", a.unwrap()), "\"special\"");
}
