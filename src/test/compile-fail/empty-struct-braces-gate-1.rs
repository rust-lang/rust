// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Feature gate test for empty struct with braces
// Can't define an empty braced struct

struct Empty1 {} //~ ERROR empty structs and enum variants with braces are unstable
struct Empty2;

enum E {
    Empty4 {}, //~ ERROR empty structs and enum variants with braces are unstable
    Empty5,
}

fn main() {
}
