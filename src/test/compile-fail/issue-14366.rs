// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let _x = "test" as &::std::any::Any;
//~^ ERROR the trait `core::kinds::Sized` is not implemented for the type `str`
//~^^ NOTE the trait `core::kinds::Sized` must be implemented for the cast to the object type
//~^^^ ERROR the trait `core::kinds::Sized` is not implemented for the type `str`
//~^^^^ NOTE the trait `core::kinds::Sized` must be implemented for the cast to the object type
}
