// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// regression test for issue #50825
// Check that the feature gate normalizes associated types.

#![allow(dead_code)]
struct Foo<T>(T);
struct Duck;
struct Quack;

trait Hello<A> where A: Animal {
}

trait Animal {
    type Noise;
}

trait Loud<R>  {
}

impl Loud<Quack> for f32 {
}

impl Animal for Duck {
    type Noise = Quack;
}

impl Hello<Duck> for Foo<f32> where f32: Loud<<Duck as Animal>::Noise> {
}

fn main() {}
