// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Zborrowck=mir

fn test_drop_replace() {
    let b: Box<isize>;
    //[mir]~^ NOTE consider changing this to `mut b`
    b = Box::new(1);    //[ast]~ NOTE first assignment
                        //[mir]~^ NOTE first assignment
    b = Box::new(2);    //[ast]~ ERROR cannot assign twice to immutable variable
                        //[mir]~^ ERROR cannot assign twice to immutable variable `b`
                        //[ast]~| NOTE cannot assign twice to immutable
                        //[mir]~| NOTE cannot assign twice to immutable
}

fn test_call() {
    let b = Box::new(1);    //[ast]~ NOTE first assignment
                            //[mir]~^ NOTE first assignment
                            //[mir]~| NOTE consider changing this to `mut b`
    b = Box::new(2);        //[ast]~ ERROR cannot assign twice to immutable variable
                            //[mir]~^ ERROR cannot assign twice to immutable variable `b`
                            //[ast]~| NOTE cannot assign twice to immutable
                            //[mir]~| NOTE cannot assign twice to immutable
}

fn test_args(b: Box<i32>) {  //[ast]~ NOTE first assignment
                                //[mir]~^ NOTE consider changing this to `mut b`
    b = Box::new(2);            //[ast]~ ERROR cannot assign twice to immutable variable
                                //[mir]~^ ERROR cannot assign to immutable argument `b`
                                //[ast]~| NOTE cannot assign twice to immutable
                                //[mir]~| NOTE cannot assign to immutable argument
}

fn main() {}
