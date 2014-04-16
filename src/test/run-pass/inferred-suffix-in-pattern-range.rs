// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let x = 2;
    let x_message = match x {
      0 .. 1     => { "not many".to_owned() }
      _          => { "lots".to_owned() }
    };
    assert_eq!(x_message, "lots".to_owned());

    let y = 2i;
    let y_message = match y {
      0 .. 1     => { "not many".to_owned() }
      _          => { "lots".to_owned() }
    };
    assert_eq!(y_message, "lots".to_owned());

    let z = 1u64;
    let z_message = match z {
      0 .. 1     => { "not many".to_owned() }
      _          => { "lots".to_owned() }
    };
    assert_eq!(z_message, "not many".to_owned());
}
