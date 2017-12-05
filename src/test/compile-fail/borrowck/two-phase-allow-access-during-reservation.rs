// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: lxl nll
//[lxl]compile-flags: -Z borrowck=mir -Z two-phase-borrows
//[nll]compile-flags: -Z borrowck=mir -Z two-phase-borrows -Z nll

// This is the second counter-example from Niko's blog post
// smallcultfollowing.com/babysteps/blog/2017/03/01/nested-method-calls-via-two-phase-borrowing/
//
// It is "artificial". It is meant to illustrate directly that we
// should allow an aliasing access during reservation, but *not* while
// the mutable borrow is active.

fn main() {
    /*0*/ let mut i = 0;

    /*1*/ let p = &mut i; // (reservation of `i` starts here)

    /*2*/ let j = i;      // OK: `i` is only reserved here

    /*3*/ *p += 1;        // (mutable borrow of `i` starts here, since `p` is used)

    /*4*/ let k = i;      //[lxl]~  ERROR cannot use `i` because it was mutably borrowed [E0503]
                          //[nll]~^ ERROR cannot use `i` because it was mutably borrowed [E0503]

    /*5*/ *p += 1;

    let _ = (j, k, p);
}
