// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo<'a> {
    i: &'a bool,
    j: Option<&'a int>,
}

impl<'a> Foo<'a> {
    fn bar(&mut self, j: &int) {
        let child = Foo {
            i: self.i,
            j: Some(j)
        };
    }
}

fn main() {}
