// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #19499. Due to incorrect caching of trait
// results for closures with upvars whose types were not fully
// computed, this rather bizarre little program (along with many more
// reasonable examples) let to ambiguity errors about not being able
// to infer sufficient type information.

fn main() {
    let n = 0;
    let it = Some(1_usize).into_iter().inspect(|_| {n;});
}
