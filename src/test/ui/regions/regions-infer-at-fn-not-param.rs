// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Parameterized1<'a> {
    g: Box<FnMut() + 'a>
}

struct NotParameterized1 {
    g: Box<FnMut() + 'static>
}

struct NotParameterized2 {
    g: Box<FnMut() + 'static>
}

fn take1<'a>(p: Parameterized1) -> Parameterized1<'a> { p }
//~^ ERROR explicit lifetime required in the type of `p`

fn take3(p: NotParameterized1) -> NotParameterized1 { p }
fn take4(p: NotParameterized2) -> NotParameterized2 { p }

fn main() {}
