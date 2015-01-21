// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unused_import_braces)]
#![allow(dead_code)]
#![allow(unused_imports)]

use test::{A}; //~ ERROR braces around A is unnecessary

mod test {
    pub struct A;
}

fn main() {}
