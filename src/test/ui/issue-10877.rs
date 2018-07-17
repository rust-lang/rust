// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo { x: isize }
extern {
    fn foo(1: ());
    //~^ ERROR: patterns aren't allowed in foreign function declarations
    fn bar((): isize);
    //~^ ERROR: patterns aren't allowed in foreign function declarations
    fn baz(Foo { x }: isize);
    //~^ ERROR: patterns aren't allowed in foreign function declarations
    fn qux((x,y): ());
    //~^ ERROR: patterns aren't allowed in foreign function declarations
    fn this_is_actually_ok(a: usize);
    fn and_so_is_this(_: usize);
}
fn main() {}
