// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn id<T>(t: T) -> T { t }

fn f<'r, T>(v: &'r T) -> &'r fn()->T { id::<&'r fn()->T>(|| *v) } //~ ERROR cannot infer an appropriate lifetime due to conflicting requirements

fn main() {
    let v = &5;
    println!("{}", f(v)());
}
