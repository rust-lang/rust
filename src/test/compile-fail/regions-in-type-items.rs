// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type item_ty_yes0 = {
    x: &uint //~ ERROR Illegal anonymous lifetime: anonymous lifetimes are not permitted here
};

type item_ty_yes1 = {
    x: &'self uint
};

type item_ty_yes2 = {
    x: &'foo uint //~ ERROR Illegal lifetime &foo: only 'self is allowed allowed as part of a type declaration
};

fn main() {}
