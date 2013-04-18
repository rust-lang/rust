// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::run;

pub fn main() {
    let cwd = os::getcwd();
    debug!("cwd = %s", cwd.to_str());
    let file = io::file_writer(&Path(~"fancy-lib/build/generated.rs"),
                               [io::Create]).get();
    file.write_str("pub fn wheeeee() { for [1, 2, 3].each() |_| { assert!(true); } }");

    // now compile the crate itself
    run::run_program("rustc", ~[~"fancy-lib/fancy-lib.rs", ~"--lib",
                                ~"-o", ~"fancy-lib/build/fancy_lib"]);
}