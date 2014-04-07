
// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs)]

use alder::*;

mod alder {
    pub enum burnside { couch, davis }
    pub enum everett { flanders, glisan, hoyt }
    pub enum irving { johnson, kearney, lovejoy }
    pub enum marshall { northrup, overton }
}

pub fn main() {
  let _pettygrove: burnside = couch;
  let _quimby: everett = flanders;
  let _raleigh: irving = johnson;
  let _savier: marshall;
}
