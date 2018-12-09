// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
enum E {
    Foo(String, String, String),
}

struct Bar {
    a: String,
    b: String,
}

fn main() {
    let bar = Bar { a: "1".to_string(), b: "2".to_string() };
    match E::Foo("".into(), "".into(), "".into()) {
        E::Foo(a, b, ref c) => {}
//~^ ERROR cannot bind by-move and by-ref in the same pattern
    }
    match bar {
        Bar {a, ref b} => {}
//~^ ERROR cannot bind by-move and by-ref in the same pattern
    }
}
