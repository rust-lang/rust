// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #36381. The trans collector was asserting that
// there are no projection types, but the `<&str as
// StreamOnce>::Position` projection contained a late-bound region,
// and we don't currently normalize in that case until the function is
// actually invoked.

pub trait StreamOnce {
    type Position;
}

impl<'a> StreamOnce for &'a str {
    type Position = usize;
}

pub fn parser<F>(_: F) {
}

fn follow(_: &str) -> <&str as StreamOnce>::Position {
    panic!()
}

fn main() {
    parser(follow);
}
