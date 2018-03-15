// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that dropck does the right thing with misc. Ty variants

use std::fmt;
struct NoisyDrop<T: fmt::Debug>(T);
impl<T: fmt::Debug> Drop for NoisyDrop<T> {
    fn drop(&mut self) {
        let _ = vec!["0wned"];
        println!("dropping {:?}", self.0)
    }
}

trait Associator {
    type As;
}
impl<T: fmt::Debug> Associator for T {
    type As = NoisyDrop<T>;
}
struct Wrap<A: Associator>(<A as Associator>::As);

fn projection() {
    let (_w, bomb);
    bomb = vec![""];
    _w = Wrap::<&[&str]>(NoisyDrop(&bomb));
}
//~^ ERROR `bomb` does not live long enough

fn closure() {
    let (_w,v);
    v = vec![""];
    _w = {
        let u = NoisyDrop(&v);
        move || u.0.len()
    };
}
//~^ ERROR `v` does not live long enough

fn main() { closure(); projection() }
