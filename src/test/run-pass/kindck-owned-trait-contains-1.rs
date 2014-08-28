// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait repeat<A> { fn get(&self) -> A; }

impl<A:Clone + 'static> repeat<A> for Box<A> {
    fn get(&self) -> A {
        (**self).clone()
    }
}

fn repeater<A:Clone + 'static>(v: Box<A>) -> Box<repeat<A>+'static> {
    box v as Box<repeat<A>+'static> // No
}

pub fn main() {
    let x = 3i;
    let y = repeater(box x);
    assert_eq!(x, y.get());
}
