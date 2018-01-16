// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(inline_attrs_in_fns_without_body)]

trait Tr {
    #[inline] //~ ERROR inline attributes have no effect on methods without bodies
    fn f1();

    #[inline(always)] //~ ERROR inline attributes have no effect on methods without bodies
    fn f2();

    #[inline]
    fn f3() {} // OK
}
