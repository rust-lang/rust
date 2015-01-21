// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait T<'a> {
    fn a(&'a self) -> &'a bool;
    fn b(&self) {
        self.a(); //~ ERROR cannot infer
    }
}

fn main() {}
