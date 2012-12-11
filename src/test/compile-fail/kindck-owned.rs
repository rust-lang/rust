// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn copy1<T: Copy>(t: T) -> fn@() -> T {
    fn@() -> T { t } //~ ERROR value may contain borrowed pointers
}

fn copy2<T: Copy Durable>(t: T) -> fn@() -> T {
    fn@() -> T { t }
}

fn main() {
    let x = &3;
    copy2(&x); //~ ERROR missing `durable`

    copy2(@3);
    copy2(@&x); //~ ERROR missing `durable`

    copy2(fn@() {});
    copy2(fn~() {}); //~ WARNING instantiating copy type parameter with a not implicitly copyable type
    copy2(fn&() {}); //~ ERROR missing `copy durable`
}
