// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Zprint-trans-items=eager

pub static FN : fn() = foo::<i32>;

pub fn foo<T>() { }

//~ TRANS_ITEM fn static_init::foo[0]<i32>
//~ TRANS_ITEM static static_init::FN[0]

fn main() { }

//~ TRANS_ITEM fn static_init::main[0]
//~ TRANS_ITEM drop-glue i8
