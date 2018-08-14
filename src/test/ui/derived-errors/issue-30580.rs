// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do not see uninformative region-related errors
// when we get some basic type-checking failure. See #30580.

pub struct Foo { a: u32 }
pub struct Pass<'a, 'tcx: 'a>(&'a mut &'a (), &'a &'tcx ());

impl<'a, 'tcx> Pass<'a, 'tcx>
{
    pub fn tcx(&self) -> &'a &'tcx () { self.1 }
    fn lol(&mut self, b: &Foo)
    {
        b.c; //~ ERROR no field `c` on type `&Foo`
        self.tcx();
    }
}

fn main() {}
