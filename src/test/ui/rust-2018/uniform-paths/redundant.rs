// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// edition:2018

use std;
use std::io;

mod foo {
    pub use std as my_std;
}

mod bar {
    pub use std::{self};
}

fn main() {
    io::stdout();
    self::std::io::stdout();
    foo::my_std::io::stdout();
    bar::std::io::stdout();
}
