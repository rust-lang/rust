// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#[feature(macro_registrar)];

extern crate syntax;

use std::any::Any;
use std::local_data;
use syntax::ast::Name;
use syntax::ext::base::SyntaxExtension;

struct Foo {
    foo: int
}

impl Drop for Foo {
    fn drop(&mut self) {}
}

#[macro_registrar]
pub fn registrar(_: |Name, SyntaxExtension|) {
    local_data_key!(foo: ~Any);
    local_data::set(foo, ~Foo { foo: 10 } as ~Any);
}

