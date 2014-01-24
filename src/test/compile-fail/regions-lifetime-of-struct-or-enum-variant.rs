// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This tests verifies that unary structs and enum variants
// are treated as rvalues and their lifetime is not bounded to
// the static scope.

struct Test;

enum MyEnum {
    Variant1
}

fn structLifetime() -> &Test {
  let testValue = &Test; //~ ERROR borrowed value does not live long enough
  testValue
}

fn variantLifetime() -> &MyEnum {
  let testValue = &Variant1; //~ ERROR borrowed value does not live long enough
  testValue
}


fn main() {}
