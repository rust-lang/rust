// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::path::{Path, PathBuf};


fn main() {
    let _tis_an_instants_play: String = "'Tis a fond Ambush—"; //~ ERROR mismatched types
    let _just_to_make_bliss: PathBuf = Path::new("/ern/her/own/surprise");
    //~^ ERROR mismatched types

    let _but_should_the_play: String = 2; // Perhaps surprisingly, we suggest .to_string() here
    //~^ ERROR mismatched types

    let _prove_piercing_earnest: Vec<usize> = &[1, 2, 3]; //~ ERROR mismatched types
}
