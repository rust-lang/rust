// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use alder::*;

mod alder {
    #[legacy_exports];
  export burnside;
  export couch;
  export everett;
  export flanders;
  export irving;
  export johnson;
  export kearney;
  export marshall;

  enum burnside { couch, davis }
  enum everett { flanders, glisan, hoyt }
  enum irving { johnson, kearney, lovejoy }
  enum marshall { northrup, overton }

}

fn main() {
  let _pettygrove: burnside = couch;
  let _quimby: everett = flanders;
  let _raleigh: irving = johnson;
  let _savier: marshall;
}
