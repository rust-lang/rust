// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Times trait

~~~ {.rust}
use iter::Times;
let ten = 10 as uint;
let mut accum = 0;
for ten.times { accum += 1; }
~~~

*/

#[allow(missing_doc)]
pub trait Times {
    fn times(&self, it: &fn() -> bool) -> bool;
}

