// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Arr0 {
    fn arr0_secret(&self);
}
trait TyParam {
    fn ty_param_secret(&self);
}

mod m {
    struct Priv;

    impl ::Arr0 for [Priv; 0] { fn arr0_secret(&self) {} }
    impl ::TyParam for Option<Priv> { fn ty_param_secret(&self) {} }
}

fn main() {
    [].arr0_secret(); //~ ERROR type `m::Priv` is private
    None.ty_param_secret(); //~ ERROR type `m::Priv` is private
}
