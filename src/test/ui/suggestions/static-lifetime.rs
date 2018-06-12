// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn _foo<'a: 'static, 'b>(_x: &'a str, _y: &'b str) -> &'a str {
//~^ WARNING unnecessary lifetime parameter `'a`
    ""
}

fn _foo1<'b, 'a: 'static>(_x: &'a str, _y: &'b str) -> &'a str {
//~^ WARNING unnecessary lifetime parameter `'a`
    ""
}

fn _foo2<'c, 'a: 'static, 'b>(_x: &'a str, _y: &'b str) -> &'a str {
//~^ WARNING unnecessary lifetime parameter `'a`
    ""
}

fn _foo3<'c, 'a: 'static, 'b, 'd>(_x: &'a str, _y: &'b str) -> &'a str {
//~^ WARNING unnecessary lifetime parameter `'a`
    ""
}

fn _foo4<'a: 'static>(_x: &'a str, _y: &str) -> &'a str {
//~^ WARNING unnecessary lifetime parameter `'a`
    ""
}

fn _foo5<'a: 'static, 'b>(_x: &'a str, _y: &'b str) -> &'b str {
//~^ WARNING unnecessary lifetime parameter `'a`
    ""
}
