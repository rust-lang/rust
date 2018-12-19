// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Documenting the kinds of lints emitted by rustdoc.
//!
//! ```
//! println!("sup");
//! ```

#![deny(rustdoc)]

/// what up, let's make an [error]
///
/// ```
/// println!("sup");
/// ```
pub fn link_error() {} //~^^^^^ ERROR cannot be resolved, ignoring it

/// wait, this doesn't have a doctest?
pub fn no_doctest() {} //~^ ERROR Missing code example in this documentation

/// wait, this *does* have a doctest?
///
/// ```
/// println!("sup");
/// ```
fn private_doctest() {} //~^^^^^ ERROR Documentation test in private item
