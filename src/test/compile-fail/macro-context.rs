// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(type_macros)]

// (typeof used because it's surprisingly hard to find an unparsed token after a stmt)
macro_rules! m {
    () => ( i ; typeof );   //~ ERROR `typeof` is a reserved keyword
                            //~| ERROR macro expansion ignores token `typeof`
                            //~| ERROR macro expansion ignores token `typeof`
                            //~| ERROR macro expansion ignores token `;`
                            //~| ERROR macro expansion ignores token `;`
                            //~| ERROR macro expansion ignores token `i`
}

m!();               //~ NOTE the usage of `m!` is likely invalid in this item context

fn main() {
    let a: m!();    //~ NOTE the usage of `m!` is likely invalid in this type context
    let i = m!();   //~ NOTE the usage of `m!` is likely invalid in this expression context
    match 0 {
        m!() => {}  //~ NOTE the usage of `m!` is likely invalid in this pattern context
    }

    m!();           //~ NOTE the usage of `m!` is likely invalid in this statement context
}
