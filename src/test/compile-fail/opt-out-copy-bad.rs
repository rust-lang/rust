// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(opt_out_copy)]

// Test that when using the `opt-out-copy` feature we still consider
// destructors to be non-movable

struct CantCopyThis;

impl Drop for CantCopyThis {
    fn drop(&mut self) { }
}

struct IWantToCopyThis {
    but_i_cant: CantCopyThis,
}

impl Copy for IWantToCopyThis {}
//~^ ERROR the trait `Copy` may not be implemented for this type

enum CantCopyThisEither {
    A,
    B(::std::kinds::marker::NoCopy),
}

enum IWantToCopyThisToo {
    ButICant(CantCopyThisEither),
}

impl Copy for IWantToCopyThisToo {}
//~^ ERROR the trait `Copy` may not be implemented for this type

fn main() {}

