// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;
use std::sync::Arc;

struct Foo(Arc<Bar>);

enum Bar {
    A(Rc<Foo>),
    B(Option<Foo>),
}

fn f<T: Send>(_: T) {}

fn main() {
    f(Foo(Arc::new(Bar::B(None))));
    //~^ ERROR E0277
    //~| ERROR E0277
}
