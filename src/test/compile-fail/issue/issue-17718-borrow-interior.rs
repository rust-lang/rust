// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S { a: usize }
static A: S  = S { a: 3 };
static B: &'static usize = &A.a;
//~^ ERROR: cannot refer to the interior of another static
static C: &'static usize = &(A.a);
//~^ ERROR: cannot refer to the interior of another static

static D: [usize; 1] = [1];
static E: usize = D[0];
//~^ ERROR: cannot refer to the interior of another static
//~^^ ERROR: cannot refer to other statics by value
static F: &'static usize = &D[0];
//~^ ERROR: cannot refer to the interior of another static

fn main() {}
