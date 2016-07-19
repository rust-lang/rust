// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {
  pub enum Enum {
    EnumStructVariant { x: u8, y: u8, z: u8 }
  }

  pub fn get_enum_struct_variant() -> () {
    Enum::EnumStructVariant { x: 1, y: 2, z: 3 }
    //~^ ERROR mismatched types
    //~| expected type `()`
    //~| found type `a::Enum`
    //~| expected (), found enum `a::Enum`
  }
}

mod b {
  mod test {
    use a;

    fn test_enum_struct_variant() {
      let enum_struct_variant = ::a::get_enum_struct_variant();
      match enum_struct_variant {
        a::Enum::EnumStructVariant { x, y, z } => {
        //~^ ERROR mismatched types
        //~| expected type `()`
        //~| found type `a::Enum`
        //~| expected (), found enum `a::Enum`
        }
      }
    }
  }
}

fn main() {}
