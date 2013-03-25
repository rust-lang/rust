// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[legacy_mode]
struct Foo<'self> {
    s: &'self str,
    u: ~()
}

pub impl<'self> Foo<'self> {
    fn get_s(&self) -> &'self str {
        self.s
    }
}

fn bar(s: &str, f: &fn(Option<Foo>)) {
    f(Some(Foo {s: s, u: ~()}));
}

fn main() {
    do bar(~"testing") |opt| {
        io::println(opt.unwrap().get_s()); //~ ERROR illegal borrow:
    };
}
