// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z emit-end-regions -Z borrowck-mir

enum Sexpression {
    Num(()),
    Cons(&'static mut Sexpression)
}

fn causes_ice(mut l: &mut Sexpression) {
    loop { match l {
        &mut Sexpression::Num(ref mut n) => {}, //[mir]~ ERROR (Mir) [E0384]
        &mut Sexpression::Cons(ref mut expr) => { //[ast]~ ERROR [E0499]
                                                  //[mir]~^ ERROR (Ast) [E0499]
                                                  //[mir]~| ERROR (Mir) [E0506]
                                                  //[mir]~| ERROR (Mir) [E0384]
                                                  //[mir]~| ERROR (Mir) [E0499]
            l = &mut **expr; //[ast]~ ERROR [E0506]
                             //[mir]~^ ERROR (Ast) [E0506]
                             //[mir]~| ERROR (Mir) [E0506]
                             //[mir]~| ERROR (Mir) [E0506]
                             //[mir]~| ERROR (Mir) [E0499]
                             //[mir]~| ERROR (Mir) [E0499]
        }
    }}
}

fn main() {
}
