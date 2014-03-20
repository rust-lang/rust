// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::kinds::marker;

struct Foo { a: int, m: marker::NoShare }

fn bar<T: Share>(_: T) {}

fn main() {
    let x = Foo { a: 5, m: marker::NoShare };
    bar(x);
    //~^ ERROR instantiating a type parameter with an incompatible type `Foo`,
    //         which does not fulfill `Share`
}
