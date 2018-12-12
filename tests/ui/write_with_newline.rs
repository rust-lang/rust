// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(clippy::write_literal)]
#![warn(clippy::write_with_newline)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // These should fail
    write!(&mut v, "Hello\n");
    write!(&mut v, "Hello {}\n", "world");
    write!(&mut v, "Hello {} {}\n", "world", "#2");
    write!(&mut v, "{}\n", 1265);

    // These should be fine
    write!(&mut v, "");
    write!(&mut v, "Hello");
    writeln!(&mut v, "Hello");
    writeln!(&mut v, "Hello\n");
    writeln!(&mut v, "Hello {}\n", "world");
    write!(&mut v, "Issue\n{}", 1265);
    write!(&mut v, "{}", 1265);
    write!(&mut v, "\n{}", 1275);
    write!(&mut v, "\n\n");
    write!(&mut v, "like eof\n\n");
    write!(&mut v, "Hello {} {}\n\n", "world", "#2");
    writeln!(&mut v, "\ndon't\nwarn\nfor\nmultiple\nnewlines\n"); // #3126
    writeln!(&mut v, "\nbla\n\n"); // #3126
}
