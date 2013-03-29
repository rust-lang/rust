// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test #2443
// exec-env:RUST_POISON_ON_FREE

fn it_takes_two(x: @int, -y: @int) -> int {
    free(y);
    debug!("about to deref");
    *x
}

fn free<T>(-_t: T) {
}

pub fn main() {
    let z = @3;
    assert!(3 == it_takes_two(z, z));
}
