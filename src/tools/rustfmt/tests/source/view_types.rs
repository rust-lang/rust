#![feature(view_types)]

use std::view::{view_projection, view_type};

type Foo_ = std::view::view_type!(Foo.{});
type Foo_ = std::view::view_type!(Foo.{ bar });
type Foo_ = std::view::view_type!(Foo.{ bar, baz });

fn f() {
    let a = view_projection!(foo.{});
    let a = view_projection!(foo.{ bar });
    let a = view_projection!(foo.{ bar, baz });
}
