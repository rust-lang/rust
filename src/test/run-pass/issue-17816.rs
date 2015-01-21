// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

fn main() {
    struct Symbol<'a, F: Fn(Vec<&'a str>) -> &'a str> { function: F }
    let f = |&: x: Vec<&str>| -> &str "foobar";
    let sym = Symbol { function: f };
    (sym.function)(vec![]);
}
