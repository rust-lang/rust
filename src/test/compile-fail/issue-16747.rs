// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait ListItem<'a> {
    fn list_name() -> &'a str;
}

struct List<'a, T: ListItem<'a>> {
//~^ ERROR the parameter type `T` may not live long enough; consider adding an explicit lifetime bo
//~^^ NOTE ...so that the reference type `&'a [T]` does not outlive the data it points at
    slice: &'a [T]
}

impl<'a, T: ListItem<'a>> Collection for List<'a, T> {
    fn len(&self) -> uint {
        0
    }
}

fn main() {}
