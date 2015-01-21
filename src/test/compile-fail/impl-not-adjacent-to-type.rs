// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub struct Foo {
        x: isize,
        y: isize,
    }
}

impl foo::Foo {
//~^ ERROR implementations may only be implemented in the same module
    fn bar() {}
}

fn main() {}

