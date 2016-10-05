// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    1.create_a_type_error[ //~ no field `create_a_type_error` on type `{integer}`
        ()+() //~ ERROR binary operation `+` cannot be applied
              //   ^ ensure that we typeck the inner expression ^
    ];
}
