// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that a trait is still object-safe (and usable) if it has
// generic methods so long as they require `Self : Sized`.


trait Counter {
    fn tick(&mut self) -> u32;
    fn with<F:FnOnce(u32)>(&self, f: F) where Self : Sized;
}

struct CCounter {
    c: u32
}

impl Counter for CCounter {
    fn tick(&mut self) -> u32 { self.c += 1; self.c }
    fn with<F:FnOnce(u32)>(&self, f: F) { f(self.c); }
}

fn tick1<C:Counter>(c: &mut C) {
    tick2(c);
    c.with(|i| ());
}

fn tick2(c: &mut Counter) {
    tick3(c);
}

fn tick3<C:?Sized+Counter>(c: &mut C) {
    c.tick();
    c.tick();
}

fn main() {
    let mut c = CCounter { c: 0 };
    tick1(&mut c);
    assert_eq!(c.tick(), 3);
}
