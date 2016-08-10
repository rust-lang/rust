// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait)]

// Helper creating a fake borrow, captured by the impl Trait.
fn borrow<'a, T>(_: &'a mut T) -> impl Copy { () }

fn stack() -> impl Copy {
    //~^ ERROR only named lifetimes are allowed in `impl Trait`
    let x = 0;
    &x
}

fn late_bound(x: &i32) -> impl Copy {
    //~^ ERROR only named lifetimes are allowed in `impl Trait`
    x
}

// FIXME(#34511) Should work but doesn't at the moment,
// region-checking needs an overhault to support this.
fn early_bound<'a>(x: &'a i32) -> impl Copy {
    //~^ ERROR only named lifetimes are allowed in `impl Trait`
    x
}

fn ambiguous<'a, 'b>(x: &'a [u32], y: &'b [u32]) -> impl Iterator<Item=u32> {
    //~^ ERROR only named lifetimes are allowed in `impl Trait`
    if x.len() < y.len() {
        x.iter().cloned()
    } else {
        y.iter().cloned()
    }
}

fn main() {}
