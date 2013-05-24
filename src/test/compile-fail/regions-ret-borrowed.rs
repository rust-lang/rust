// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that you cannot use generic types to return a region outside
// of its bound.  Here, in the `return_it()` fn, we call with() but
// with R bound to &int from the return_it.  Meanwhile, with()
// provides a value that is only good within its own stack frame. This
// used to successfully compile because we failed to account for the
// fact that fn(x: &int) rebound the region &.

fn with<R>(f: &fn(x: &int) -> R) -> R {
    f(&3)
}

fn return_it() -> &int {
    with(|o| o) //~ ERROR mismatched types
        //~^ ERROR lifetime of return value does not outlive the function call
        //~^^ ERROR cannot infer an appropriate lifetime
}

fn main() {
    let x = return_it();
    debug!("foo=%d", *x);
}
