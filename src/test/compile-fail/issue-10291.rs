// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

fn test<'x>(x: &'x isize) {
    drop::<Box<for<'z> FnMut(&'z isize) -> &'z isize>>(box |z| {
        x
        //~^ ERROR cannot infer an appropriate lifetime
    });
}

fn main() {}
