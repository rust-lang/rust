// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct ret;
struct obj;

impl obj {
    fn func() -> ret {
        ret
    }
}

fn func() -> ret {
    ret
}

fn main() {
    obj::func.x();
    //~^ ERROR no method named `x` found for type `fn() -> ret {obj::func}` in the current scope
    //~^^ NOTE obj::func is a function, perhaps you wish to call it
    func.x();
    //~^ ERROR no method named `x` found for type `fn() -> ret {func}` in the current scope
    //~^^ NOTE func is a function, perhaps you wish to call it
}
