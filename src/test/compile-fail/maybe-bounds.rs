// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Tr: ?Sized {} //~ ERROR `?Trait` is not permitted in supertraits
                    //~^ NOTE traits are `?Sized` by default

type A1 = Tr + ?Sized; //~ ERROR `?Trait` is not permitted in trait object types
type A2 = for<'a> Tr + ?Sized; //~ ERROR `?Trait` is not permitted in trait object types

fn main() {}
