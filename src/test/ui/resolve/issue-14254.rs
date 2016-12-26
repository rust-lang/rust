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
    fn bar(&self);
    fn baz(&self) { }
    fn bah(_: Option<&Self>) { }
}

struct BarTy {
    x : isize,
    y : f64,
}

impl BarTy {
    fn a() {}
    fn b(&self) {}
}

impl Foo for *const BarTy {
    fn bar(&self) {
        baz();
        //~^ ERROR unresolved function `baz`
        //~| NOTE did you mean `self.baz(...)`?
        a;
        //~^ ERROR unresolved value `a`
        //~| NOTE no resolution found
    }
}

impl<'a> Foo for &'a BarTy {
    fn bar(&self) {
        baz();
        //~^ ERROR unresolved function `baz`
        //~| NOTE did you mean `self.baz(...)`?
        x;
        //~^ ERROR unresolved value `x`
        //~| NOTE did you mean `self.x`?
        y;
        //~^ ERROR unresolved value `y`
        //~| NOTE did you mean `self.y`?
        a;
        //~^ ERROR unresolved value `a`
        //~| NOTE no resolution found
        bah;
        //~^ ERROR unresolved value `bah`
        //~| NOTE did you mean `Self::bah`?
        b;
        //~^ ERROR unresolved value `b`
        //~| NOTE no resolution found
    }
}

impl<'a> Foo for &'a mut BarTy {
    fn bar(&self) {
        baz();
        //~^ ERROR unresolved function `baz`
        //~| NOTE did you mean `self.baz(...)`?
        x;
        //~^ ERROR unresolved value `x`
        //~| NOTE did you mean `self.x`?
        y;
        //~^ ERROR unresolved value `y`
        //~| NOTE did you mean `self.y`?
        a;
        //~^ ERROR unresolved value `a`
        //~| NOTE no resolution found
        bah;
        //~^ ERROR unresolved value `bah`
        //~| NOTE did you mean `Self::bah`?
        b;
        //~^ ERROR unresolved value `b`
        //~| NOTE no resolution found
    }
}

impl Foo for Box<BarTy> {
    fn bar(&self) {
        baz();
        //~^ ERROR unresolved function `baz`
        //~| NOTE did you mean `self.baz(...)`?
        bah;
        //~^ ERROR unresolved value `bah`
        //~| NOTE did you mean `Self::bah`?
    }
}

impl Foo for *const isize {
    fn bar(&self) {
        baz();
        //~^ ERROR unresolved function `baz`
        //~| NOTE did you mean `self.baz(...)`?
        bah;
        //~^ ERROR unresolved value `bah`
        //~| NOTE did you mean `Self::bah`?
    }
}

impl<'a> Foo for &'a isize {
    fn bar(&self) {
        baz();
        //~^ ERROR unresolved function `baz`
        //~| NOTE did you mean `self.baz(...)`?
        bah;
        //~^ ERROR unresolved value `bah`
        //~| NOTE did you mean `Self::bah`?
    }
}

impl<'a> Foo for &'a mut isize {
    fn bar(&self) {
        baz();
        //~^ ERROR unresolved function `baz`
        //~| NOTE did you mean `self.baz(...)`?
        bah;
        //~^ ERROR unresolved value `bah`
        //~| NOTE did you mean `Self::bah`?
    }
}

impl Foo for Box<isize> {
    fn bar(&self) {
        baz();
        //~^ ERROR unresolved function `baz`
        //~| NOTE did you mean `self.baz(...)`?
        bah;
        //~^ ERROR unresolved value `bah`
        //~| NOTE did you mean `Self::bah`?
    }
}
