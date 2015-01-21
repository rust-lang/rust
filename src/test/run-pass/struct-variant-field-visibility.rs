// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub enum Foo {
        Bar { a: int }
    }
}

fn f(f: foo::Foo) {
    match f {
        foo::Foo::Bar { a: _a } => {}
    }
}

pub fn main() {}
