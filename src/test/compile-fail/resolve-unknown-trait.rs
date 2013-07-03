// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait NewTrait : SomeNonExistentTrait {}
//~^ ERROR attempt to derive a nonexistent trait `SomeNonExistentTrait`

impl SomeNonExistentTrait for int {}
//~^ ERROR attempt to implement a nonexistent trait `SomeNonExistentTrait`

fn f<T:SomeNonExistentTrait>() {}
//~^ ERROR attempt to bound type parameter with a nonexistent trait `SomeNonExistentTrait`

