//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:pin-ergonomics-hir.pp

#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]
#[attr = MacroUse {arguments: UseAll}]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

use std::pin::Pin;

struct Foo;

impl Foo {
    fn baz(&mut self) { }

    fn baz_const(&self) { }

    fn baz_lt<'a>(&mut self) { }

    fn baz_const_lt(&self) { }
}

fn foo(_: Pin<&'_ mut Foo>) { }
fn foo_lt<'a>(_: Pin<&'a mut Foo>) { }

fn foo_const(_: Pin<&'_ Foo>) { }
fn foo_const_lt(_: Pin<&'_ Foo>) { }

fn bar() {
    let mut x: Pin<&mut _> = &pin mut Foo;
    foo(x.as_mut());
    foo(x.as_mut());
    foo_const(x);

    let x: Pin<&_> = &pin const Foo;

    foo_const(x);
    foo_const(x);
}

fn main() { }
