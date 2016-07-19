// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Trait1<'l0, T0> {}
trait Trait0<'l0>  {}

impl <'l0, 'l1, T0> Trait1<'l0, T0> for bool where T0 : Trait0<'l0>, T0 : Trait0<'l1> {}
//~^ ERROR type annotations required: cannot resolve `T0: Trait0<'l0>`
//~^^ NOTE required by `Trait0`

fn main() {}
