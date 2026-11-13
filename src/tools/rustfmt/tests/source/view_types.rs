#![feature(view_types, view_type_macro)]

use std::view::view_type;

type Foo_ = std::view::view_type!(Foo.{});
type Foo_ = std::view::view_type!(Foo.{ bar });
type Foo_ = std::view::view_type!(Foo.{ bar, baz });
