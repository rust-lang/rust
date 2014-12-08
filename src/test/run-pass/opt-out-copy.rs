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

// Test the opt-out-copy feature guard. This is the same as the
// "opt-in-copy.rs" test from compile-fail, except that it is using
// the feature guard, and hence the structureds in this file are
// implicitly copyable, and hence we get no errors. This test can be
// safely removed once the opt-out-copy "feature" is rejected.

struct CantCopyThis;

struct IWantToCopyThis {
    but_i_cant: CantCopyThis,
}

impl Copy for IWantToCopyThis {}

enum CantCopyThisEither {
    A,
    B,
}

enum IWantToCopyThisToo {
    ButICant(CantCopyThisEither),
}

impl Copy for IWantToCopyThisToo {}

fn is_copy<T:Copy>() { }

fn main() {
    is_copy::<CantCopyThis>();
    is_copy::<CantCopyThisEither>();
    is_copy::<IWantToCopyThis>();
    is_copy::<IWantToCopyThisToo>();
}

