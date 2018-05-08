// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_let)]

fn main() {}

struct FakeNeedsDrop;

impl Drop for FakeNeedsDrop {
    fn drop(&mut self) {}
}

// ok
const X: FakeNeedsDrop = { let x = FakeNeedsDrop; x };

// error
const Y: FakeNeedsDrop = { let mut x = FakeNeedsDrop; x = FakeNeedsDrop; x };
//~^ ERROR constant contains unimplemented expression type

// error
const Z: () = { let mut x = None; x = Some(FakeNeedsDrop); };
//~^ ERROR constant contains unimplemented expression type
