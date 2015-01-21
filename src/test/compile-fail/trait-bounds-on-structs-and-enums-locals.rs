// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Trait {}

struct Foo<T:Trait> {
    x: T,
}

fn main() {
    let foo = Foo {
    //~^ ERROR not implemented
        x: 3is
    };

    let baz: Foo<usize> = panic!();
    //~^ ERROR not implemented
}

