// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

trait Foo {
    fn bar(&self);
    fn baz(&self) { }
    fn bah(_: Option<Self>) { }
}

struct BarTy {
    x : isize,
    y : f64,
}

impl BarTy {
    fn a() {}
    fn b(&self) {}
}

// If these fail, it's necessary to update rustc_resolve and the cfail tests.
impl Foo for *const BarTy {
    fn bar(&self) {
        self.baz();
        BarTy::a();
        Foo::bah(None::<*const BarTy>);
    }
}

// If these fail, it's necessary to update rustc_resolve and the cfail tests.
impl<'a> Foo for &'a BarTy {
    fn bar(&self) {
        self.baz();
        self.x;
        self.y;
        BarTy::a();
        Foo::bah(None::<&BarTy>);
        self.b();
    }
}

// If these fail, it's necessary to update rustc_resolve and the cfail tests.
impl<'a> Foo for &'a mut BarTy {
    fn bar(&self) {
        self.baz();
        self.x;
        self.y;
        BarTy::a();
        Foo::bah(None::<&mut BarTy>);
        self.b();
    }
}

// If these fail, it's necessary to update rustc_resolve and the cfail tests.
impl Foo for Box<BarTy> {
    fn bar(&self) {
        self.baz();
        Foo::bah(None::<Box<BarTy>>);
    }
}

// If these fail, it's necessary to update rustc_resolve and the cfail tests.
impl Foo for *const isize {
    fn bar(&self) {
        self.baz();
        Foo::bah(None::<*const isize>);
    }
}

// If these fail, it's necessary to update rustc_resolve and the cfail tests.
impl<'a> Foo for &'a isize {
    fn bar(&self) {
        self.baz();
        Foo::bah(None::<&isize>);
    }
}

// If these fail, it's necessary to update rustc_resolve and the cfail tests.
impl<'a> Foo for &'a mut isize {
    fn bar(&self) {
        self.baz();
        Foo::bah(None::<&mut isize>);
    }
}

// If these fail, it's necessary to update rustc_resolve and the cfail tests.
impl Foo for Box<isize> {
    fn bar(&self) {
        self.baz();
        Foo::bah(None::<Box<isize>>);
    }
}

fn main() {}
