// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn expr_method_call_19() {
    struct S19 { x: int }
    impl S19 { fn inner(self) -> S19 { S19 { x: self.x + self.x } } }
    let s = S19 { x: 19 };
    s.inner().inner();
}
