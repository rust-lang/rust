// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that parentheses form doesn't work in expression paths.

struct Bar<A,R> {
    f: A, r: R
}

impl<A,B> Bar<A,B> {
    fn new() -> Bar<A,B> { panic!() }
}

fn bar() {
    let b = Box::Bar::<isize,usize>::new(); // OK

    let b = Box::Bar::()::new();
    //~^ ERROR expected ident, found `(`
}

fn main() { }

