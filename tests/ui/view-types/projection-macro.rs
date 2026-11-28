//@ revisions: legal field_mismatch unknown_field
//@[legal] run-pass

#![feature(view_types)]
#![allow(unused)]

use std::ops::Deref;
use std::view::{view_projection, view_type};

struct Foo {
    a: (),
    b: (),
}

struct Bar;

impl Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        todo!()
    }
}

#[cfg(legal)]
fn legal() {
    let foo = Foo { a: (), b: () };
    let foo_ref = &foo;
    let bar = Bar;

    let _foo_a: &view_type!(Foo.{ a }) = &view_projection!(foo.{ a });
    let _through_deref: &view_type!(Foo.{ a }) = &view_projection!(foo_ref.{ a });
    let _through_deref_impl: &view_type!(Foo.{ a }) = &view_projection!(bar.{ a });
    let _inferred = &view_projection!(foo.{ a });
}

#[cfg(field_mismatch)]
fn field_mismatch() {
    let foo = Foo { a: (), b: () };
    let _field_mismatch: &view_type!(Foo.{ a }) = &view_projection!(foo.{ b });
    //[field_mismatch]~^ ERROR mismatched type
}

#[cfg(unknown_field)]
fn unknown_field() {
    let foo = Foo { a: (), b: () };

    let _unknown_field = &view_projection!(foo.{ c });
    //[unknown_field]~^ ERROR field `c` cannot be viewed from type `Foo`
}

fn main() {}
