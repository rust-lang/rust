// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! id {
    ($s: pat) => ($s);
}

fn main() {
    match (Some(123), Some(456)) {
        (id!(Some(a)), _) | (_, id!(Some(a))) => println!("{}", a),
        _ => (),
    }
}
