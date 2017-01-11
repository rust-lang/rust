// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(pub_restricted)]

enum E {}
trait Tr {}

pub(E) struct S; //~ ERROR expected module, found enum `E`
pub(Tr) struct Z; //~ ERROR expected module, found trait `Tr`
pub(std::vec) struct F; //~ ERROR visibilities can only be restricted to ancestor modules
pub(nonexistent) struct G; //~ ERROR cannot find module `nonexistent` in the crate root
pub(too_soon) struct H; //~ ERROR cannot find module `too_soon` in the crate root

// Visibilities are resolved eagerly without waiting for modules becoming fully populated.
// Visibilities can only use ancestor modules legally which are always available in time,
// so the worst thing that can happen due to eager resolution is a suboptimal error message.
mod too_soon {}

fn main () {}
