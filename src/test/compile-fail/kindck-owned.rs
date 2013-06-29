// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn copy1<T:Copy>(t: T) -> @fn() -> T {
    let result: @fn() -> T = || copy t; //~ ERROR does not fulfill `'static`
    result
}

fn copy2<T:Copy + 'static>(t: T) -> @fn() -> T {
    let result: @fn() -> T = || copy t;
    result
}

fn main() {
    let x = &3;
    copy2(&x); //~ ERROR does not fulfill `'static`

    copy2(@3);
    copy2(@&x); //~ ERROR does not fulfill `'static`

    let boxed: @fn() = || {};
    copy2(boxed);
    let owned: ~fn() = || {};
    copy2(owned);    //~ ERROR does not fulfill `Copy`
    let borrowed: &fn:Copy() = || {};
    copy2(borrowed); //~ ERROR does not fulfill `'static`
}
