// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// regression test for issue #41498.

struct S;
impl S {
    fn mutate(&mut self) {}
}

fn call_and_ref<T, F: FnOnce() -> T>(x: &mut Option<T>, f: F) -> &mut T {
    *x = Some(f());
    x.as_mut().unwrap()
}

fn main() {
    let mut n = None;
    call_and_ref(&mut n, || [S])[0].mutate();
}
