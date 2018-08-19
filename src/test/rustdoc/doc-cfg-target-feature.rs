// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// only-x86_64
// compile-flags:--test
// should-fail
// no-system-llvm

// #49723: rustdoc didn't add target features when extracting or running doctests

#![feature(doc_cfg)]

/// Foo
///
/// # Examples
///
/// ```
/// #![feature(cfg_target_feature)]
///
/// #[cfg(target_feature = "sse")]
/// assert!(false);
/// ```
#[doc(cfg(target_feature = "sse"))]
pub unsafe fn foo() {}
