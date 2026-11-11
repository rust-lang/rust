//@ check-fail
#![feature(extern_types, sized_hierarchy)]

use std::marker::{MetaSized, PointeeSized};

fn bare<T>() {}


fn sized<T: Sized>() {}

fn neg_sized<T: ?Sized>() {}


fn metasized<T: MetaSized>() {}

fn neg_metasized<T: ?MetaSized>() {}
//~^ ERROR bound modifier `?` can only be applied to `Sized`


fn pointeesized<T: PointeeSized>() { }

fn neg_pointeesized<T: ?PointeeSized>() { }
//~^ ERROR bound modifier `?` can only be applied to `Sized`


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
