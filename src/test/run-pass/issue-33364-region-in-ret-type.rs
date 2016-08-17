// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #33364: While working on the hack/fix to workaround an ICE, I
// injected a new ICE that was not tested by our run-pass test suite
// (but was exposed by attempting to bootstrap the compiler itself, as
// well as our compile-fail tests). This test attempts to codify the
// problem I encountered at that time, as a run-pass test.

#![feature(unboxed_closures)]
fn main()
{
    static X: u32 = 3;
    fn lifetime_in_ret_type_alone<'a>() -> &'a u32 { &X }
    fn apply_thunk<T: FnOnce<()>>(t: T, _x: T::Output) { t(); }

    apply_thunk(lifetime_in_ret_type_alone, &3);
}
