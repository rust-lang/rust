// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can derive lifetime bounds on `Self` from trait
// inheritance.

trait Static : 'static { }

trait Is<'a> : 'a { }

struct Inv<'a> {
    x: Option<&'a mut &'a int>
}

fn check_bound<'a,A:'a>(x: Inv<'a>, a: A) { }

// In these case, `Self` inherits `'static`.

trait InheritsFromStatic : 'static {
    fn foo1<'a>(self, x: Inv<'a>) {
        check_bound(x, self)
    }
}
trait InheritsFromStaticIndirectly : Static {
    fn foo1<'a>(self, x: Inv<'a>) {
        check_bound(x, self)
    }
}


// In these case, `Self` inherits `'a`.

trait InheritsFromIs<'a> : 'a {
    fn foo(self, x: Inv<'a>) {
        check_bound(x, self)
    }
}

trait InheritsFromIsIndirectly<'a> : Is<'a> {
    fn foo(self, x: Inv<'a>) {
        check_bound(x, self)
    }
}

// In this case, `Self` inherits nothing.

trait InheritsFromNothing<'a> {
    fn foo(self, x: Inv<'a>) {
        check_bound(x, self)
            //~^ ERROR parameter type `Self` may not live long enough
    }
}

fn main() { }
