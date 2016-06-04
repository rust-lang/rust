// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The pointer-sized signed integer type.
//!
//! *[See also the `isize` primitive type](../../std/primitive.isize.html).*

#![stable(feature = "rust1", since = "1.0.0")]

#[cfg(target_pointer_width = "16")]
int_module! { isize, 16 }
#[cfg(target_pointer_width = "32")]
int_module! { isize, 32 }
#[cfg(target_pointer_width = "64")]
int_module! { isize, 64 }
