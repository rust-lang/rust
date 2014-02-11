// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test
// ignore'd due to problems with by-value self.

// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait get_ctxt<'a> {
    fn get_ctxt(self) -> &'a uint;
}

fn make_gc1(gc: @get_ctxt<'a>) -> @get_ctxt<'b>  {
    return gc; //~ ERROR mismatched types: expected `@get_ctxt/&b` but found `@get_ctxt/&a`
}

struct Foo {
    r: &'a uint
}

impl get_ctxt for Foo<'a> {
    fn get_ctxt(&self) -> &'a uint { self.r }
}

fn make_gc2<'a,'b>(foo: Foo<'a>) -> @get_ctxt<'b>  {
    return @foo as @get_ctxt; //~ ERROR cannot infer
}

fn main() {
}
