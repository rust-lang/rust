// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fs::OpenOptions;

#[allow(unused_must_use)]
#[warn(clippy::nonsensical_open_options)]
fn main() {
    OpenOptions::new().read(true).truncate(true).open("foo.txt");
    OpenOptions::new().append(true).truncate(true).open("foo.txt");

    OpenOptions::new().read(true).read(false).open("foo.txt");
    OpenOptions::new().create(true).create(false).open("foo.txt");
    OpenOptions::new().write(true).write(false).open("foo.txt");
    OpenOptions::new().append(true).append(false).open("foo.txt");
    OpenOptions::new().truncate(true).truncate(false).open("foo.txt");
}
