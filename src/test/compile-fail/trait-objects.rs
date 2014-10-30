// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn foo(self);
}

trait Bar {
    fn bar(&self, x: &Self);
}

trait Baz {
    fn baz<T>(&self, x: &T);
}

impl Foo for int {
    fn foo(self) {}
}

impl Bar for int {
    fn bar(&self, _x: &int) {}
}

impl Baz for int {
    fn baz<T>(&self, _x: &T) {}
}

fn main() {
    let _: &Foo = &42i; //~ ERROR cannot convert to a trait object
    let _: &Bar = &42i; //~ ERROR cannot convert to a trait object
    let _: &Baz = &42i; //~ ERROR cannot convert to a trait object

    let _ = &42i as &Foo; //~ ERROR cannot convert to a trait object
    let _ = &42i as &Bar; //~ ERROR cannot convert to a trait object
    let _ = &42i as &Baz; //~ ERROR cannot convert to a trait object
}
