// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Common code used for tests that model the Fn/FnMut/FnOnce hierarchy.

pub trait Go {
    fn go(&self, arg: int);
}

pub fn go<G:Go>(this: &G, arg: int) {
    this.go(arg)
}

pub trait GoMut {
    fn go_mut(&mut self, arg: int);
}

pub fn go_mut<G:GoMut>(this: &mut G, arg: int) {
    this.go_mut(arg)
}

pub trait GoOnce {
    fn go_once(self, arg: int);
}

pub fn go_once<G:GoOnce>(this: G, arg: int) {
    this.go_once(arg)
}

impl<G> GoMut for G
    where G : Go
{
    fn go_mut(&mut self, arg: int) {
        go(&*self, arg)
    }
}

impl<G> GoOnce for G
    where G : GoMut
{
    fn go_once(mut self, arg: int) {
        go_mut(&mut self, arg)
    }
}
