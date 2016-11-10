// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(fn_traits)]

fn id<T>(t: T) -> T { t }

fn f<'r, T>(v: &'r T) -> Box<FnMut() -> T + 'r> {
    // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
    id(Box::new(|| *v))
        //~^ ERROR E0373
        //~| NOTE `v` is borrowed here
        //~| NOTE may outlive borrowed value `v`
        //~| ERROR E0507
        //~| NOTE cannot move out of borrowed content
}

fn main() {
    let v = &5;
    println!("{}", f(v).call_mut(()));
}
