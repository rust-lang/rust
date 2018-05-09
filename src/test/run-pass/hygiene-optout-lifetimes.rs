// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(decl_macro)]
#![feature(macro_hygiene_optout)]

use std::marker::PhantomData;

macro m($lifetime:tt) {
    pub struct #Foo<$lifetime>(PhantomData<&#'a ()>);
}

fn main() {
    m!('a);
    let _ = Foo(Default::default());
}
