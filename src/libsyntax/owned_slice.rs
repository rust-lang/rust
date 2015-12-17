// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// A non-growable owned slice.
#[unstable(feature = "rustc_private", issue = "0")]
#[rustc_deprecated(since = "1.7.0", reason = "use `ptr::P<[T]>` instead")]
pub type OwnedSlice<T> = ::ptr::P<[T]>;
