// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=compare

pub fn main(){
    let maybe = Some(vec![true, true]);

    loop {
        if let Some(thing) = maybe {
        }
        //~^^ ERROR use of partially moved value: `maybe` (Ast) [E0382]
        //~| ERROR use of moved value: `(maybe as std::prelude::v1::Some).0` (Ast) [E0382]
        //~| ERROR use of moved value: `maybe` (Mir) [E0382]
        //~| ERROR use of moved value: `maybe` (Mir) [E0382]
        //~| ERROR use of moved value (Mir) [E0382]
        //~| ERROR borrow of moved value: `maybe` (Mir) [E0382]
    }
}
