// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Because this module is temporary...
#![allow(missing_docs)]
#![unstable(feature = "std_misc")]

use alloc::boxed::{Box, FnBox};
use core::marker::Send;

pub type Thunk<'a, A=(), R=()> =
    Box<FnBox<A,Output=R> + Send + 'a>;

