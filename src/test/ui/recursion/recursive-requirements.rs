// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker::PhantomData;

struct AssertSync<T: Sync>(PhantomData<T>);

pub struct Foo {
    bar: *const Bar,
    phantom: PhantomData<Bar>,
}

pub struct Bar {
    foo: *const Foo,
    phantom: PhantomData<Foo>,
}

fn main() {
    let _: AssertSync<Foo> = unimplemented!(); //~ ERROR E0275
}
