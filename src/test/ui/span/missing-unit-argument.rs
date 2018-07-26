// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo(():(), ():()) {}
fn bar(():()) {}

struct S;
impl S {
    fn baz(self, (): ()) { }
    fn generic<T>(self, _: T) { }
}

fn main() {
    let _: Result<(), String> = Ok(); //~ ERROR this function takes
    foo(); //~ ERROR this function takes
    foo(()); //~ ERROR this function takes
    bar(); //~ ERROR this function takes
    S.baz(); //~ ERROR this function takes
    S.generic::<()>(); //~ ERROR this function takes
}
