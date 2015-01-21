// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[repr(packed)]
#[derive(PartialEq, Show)]
struct Foo {
    a: i8,
    b: i16,
    c: i8
}

impl Copy for Foo {}

#[link(name = "test", kind = "static")]
extern {
    fn foo(f: Foo) -> Foo;
}

fn main() {
    unsafe {
        let a = Foo { a: 1, b: 2, c: 3 };
        let b = foo(a);
        assert_eq!(a, b);
    }
}
