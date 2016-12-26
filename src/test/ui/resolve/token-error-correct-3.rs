// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do some basic error correcton in the tokeniser (and don't spew
// too many bogus errors).

pub mod raw {
    use std::{io, fs};
    use std::path::Path;

    pub fn ensure_dir_exists<P: AsRef<Path>, F: FnOnce(&Path)>(path: P,
                                                               callback: F)
                                                               -> io::Result<bool> {
        if !is_directory(path.as_ref()) { //~ ERROR: unresolved function `is_directory`
                                          //~^ NOTE: no resolution found
            callback(path.as_ref();  //~ NOTE: unclosed delimiter
                     //~^ ERROR: expected one of
            fs::create_dir_all(path.as_ref()).map(|()| true) //~ ERROR: mismatched types
            //~^ expected (), found enum `std::result::Result`
            //~| expected type `()`
            //~| found type `std::result::Result<bool, std::io::Error>`
        } else { //~ ERROR: incorrect close delimiter: `}`
            //~^ ERROR: expected one of
            Ok(false);
        }

        panic!();
    }
}

fn main() {}
