#![feature(view_types, sized_hierarchy)]

use std::marker::{MetaSized, PointeeSized, Sized};
use std::view::view_type;

struct A {
    b: (),
    c: (),
}

fn assert_sized<T>() {}
fn assert_metasized<T: MetaSized>() {}
fn assert_pointeesized<T: PointeeSized>() {}

fn main() {
    assert_sized::<view_type!(A.{})>();
    //~^ ERROR the size for values of type `view_type!(A.{ () })` cannot be known at compilation time
    assert_metasized::<view_type!(A.{})>();
    //~^ ERROR the size for values of type `view_type!(A.{ () })` cannot be known
    assert_pointeesized::<view_type!(A.{})>();

    // View types stay unsized even if they borrow some/all the fields.
    assert_sized::<view_type!(A.{ b })>();
    //~^ ERROR the size for values of type `view_type!(A.{ (field_of!(A, b),) })` cannot be known at compilation time
    assert_metasized::<view_type!(A.{ b })>();
    //~^ ERROR the size for values of type `view_type!(A.{ (field_of!(A, b),) })` cannot be known
    assert_pointeesized::<view_type!(A.{ b })>();

    assert_sized::<view_type!(A.{ b, c })>();
    //~^ ERROR the size for values of type `view_type!(A.{ (field_of!(A, b), field_of!(A, c)) })` cannot be known at compilation time
    assert_metasized::<view_type!(A.{ b, c })>();
    //~^ ERROR the size for values of type `view_type!(A.{ (field_of!(A, b), field_of!(A, c)) })` cannot be known
    assert_pointeesized::<view_type!(A.{ b, c })>();
}
