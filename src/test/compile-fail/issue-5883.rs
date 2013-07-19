// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;

struct Struct {
    r: io::Reader //~ ERROR reference to trait `io::Reader` where a type is expected
}

fn new_struct(r: io::Reader) -> Struct { //~ ERROR reference to trait `io::Reader` where a type is expected
    Struct { r: r }
}

trait Curve {}
enum E {X(Curve)} //~ ERROR reference to trait `Curve` where a type is expected
fn main() {}
