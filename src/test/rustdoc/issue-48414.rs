// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-48414.rs

// ICE when resolving paths for a trait that linked to another trait, when both were in an external
// crate

#![crate_name = "base"]

extern crate issue_48414;

#[doc(inline)]
pub use issue_48414::{SomeTrait, OtherTrait};
