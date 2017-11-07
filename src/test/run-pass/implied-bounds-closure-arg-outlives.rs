// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to handle the relationships between free
// regions bound in a closure callback.

#[derive(Copy, Clone)]
struct MyCx<'short, 'long: 'short> {
    short: &'short u32,
    long: &'long u32,
}

impl<'short, 'long> MyCx<'short, 'long> {
    fn short(self) -> &'short u32 { self.short }
    fn long(self) -> &'long u32 { self.long }
    fn set_short(&mut self, v: &'short u32) { self.short = v; }
}

fn with<F, R>(op: F) -> R
where
    F: for<'short, 'long> FnOnce(MyCx<'short, 'long>) -> R,
{
    op(MyCx {
        short: &22,
        long: &22,
    })
}

fn main() {
    with(|mut cx| {
        // For this to type-check, we need to be able to deduce that
        // the lifetime of `l` can be `'short`, even though it has
        // input from `'long`.
        let l = if true { cx.long() } else { cx.short() };
        cx.set_short(l);
    });
}
