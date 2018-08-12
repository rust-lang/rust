// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Clone, Copy)]
struct S;

trait T {
    fn foo((x, y): (i32, i32)); //~ ERROR patterns aren't allowed in methods without bodies

    fn bar((x, y): (i32, i32)) {} //~ ERROR patterns aren't allowed in methods without bodies

    fn f(&ident: &S) {} // ok
    fn g(&&ident: &&S) {} // ok
    fn h(mut ident: S) {} // ok
}

fn main() {}
