// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #49685: drop elaboration was not revealing the
// value of `impl Trait` returns, leading to an ICE.

fn main() {
    let _ = Some(())
        .into_iter()
        .flat_map(|_| Some(()).into_iter().flat_map(func));
}

fn func(_: ()) -> impl Iterator<Item = ()> {
    Some(()).into_iter().flat_map(|_| vec![])
}
