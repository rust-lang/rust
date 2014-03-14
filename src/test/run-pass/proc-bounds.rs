// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo<T>() {}
fn bar<T>(_: T) {}

fn is_send<T: Send>() {}
fn is_freeze<T: Freeze>() {}
fn is_static<T: 'static>() {}

pub fn main() {
    foo::<proc()>();
    foo::<proc:()>();
    foo::<proc:Send()>();
    foo::<proc:Send + Freeze()>();
    foo::<proc:'static + Send + Freeze()>();

    is_send::<proc:Send()>();
    is_freeze::<proc:Freeze()>();
    is_static::<proc:'static()>();


    let a = 3;
    bar::<proc:()>(proc() {
        let b = &a;
        println!("{}", *b);
    });
}
