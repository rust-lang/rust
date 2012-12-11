// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct list<T> {
    element: &self/T,
    mut next: Option<@list<T>>
}

impl<T> list<T>{
    fn addEnd(&self, element: &self/T) {
        let newList = list {
            element: element,
            next: option::None
        };

        self.next = Some(@(move newList));
    }
}

fn main() {
    let s = @"str";
    let ls = list {
        element: &s,
        next: option::None
    };
    io::println(*ls.element);
}
