//@ check-fail
//@ compile-flags: -Zunstable-options
//@ edition: future
#![allow(sized_hierarchy_migration)]
#![feature(extern_types, sized_hierarchy)]

use std::marker::{MetaSized, PointeeSized};

fn bare<T>() {}


fn sized<T: Sized>() {}

fn neg_sized<T: ?Sized>() {}


fn metasized<T: MetaSized>() {}

fn neg_metasized<T: ?MetaSized>() {}
//~^ ERROR relaxing a default bound only does something for `?Sized`


fn pointeesized<T: PointeeSized>() { }

fn neg_pointeesized<T: ?PointeeSized>() { }
//~^ ERROR relaxing a default bound only does something for `?Sized`


fn main() {
    // Functions which should have a `T: Sized` bound - check for an error given a non-Sized type:

    bare::<[u8]>();
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    sized::<[u8]>();
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    metasized::<[u8]>();
    pointeesized::<[u8]>();

    // Functions which should have a `T: MetaSized` bound - check for an error given a
    // non-MetaSized type:
    unsafe extern "C" {
        type Foo;
    }

    bare::<Foo>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known
    sized::<Foo>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known
    metasized::<Foo>();
    //~^ ERROR the size for values of type `main::Foo` cannot be known
    pointeesized::<Foo>();
}
