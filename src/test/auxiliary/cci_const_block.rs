// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub static BLOCK_FN_DEF: fn(uint) -> uint = {
    fn foo(a: uint) -> uint {
        a + 10
    }
    foo
};
