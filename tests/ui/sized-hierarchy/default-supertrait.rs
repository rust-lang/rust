//@ check-fail
#![feature(sized_hierarchy)]

use std::marker::{MetaSized, PointeeSized};

trait Sized_: Sized { }

trait NegSized: ?Sized { }
//~^ ERROR `?Trait` is not permitted in supertraits

trait MetaSized_: MetaSized { }

trait NegMetaSized: ?MetaSized { }
//~^ ERROR `?Trait` is not permitted in supertraits


trait PointeeSized_: PointeeSized { }

trait NegPointeeSized: ?PointeeSized { }
//~^ ERROR `?Trait` is not permitted in supertraits

trait Bare {}

fn requires_sized<T: Sized>() {}
fn requires_metasized<T: MetaSized>() {}
fn requires_pointeesized<T: PointeeSized>() {}

fn with_sized_supertrait<T: PointeeSized + Sized_>() {
    requires_sized::<T>();
    requires_metasized::<T>();
    requires_pointeesized::<T>();
}

fn with_metasized_supertrait<T: PointeeSized + MetaSized_>() {
    requires_sized::<T>();
    //~^ ERROR the size for values of type `T` cannot be known at compilation time
    requires_metasized::<T>();
    requires_pointeesized::<T>();
}

// It isn't really possible to write this one..
fn with_pointeesized_supertrait<T: PointeeSized + PointeeSized_>() {
    requires_sized::<T>();
    //~^ ERROR the size for values of type `T` cannot be known
    requires_metasized::<T>();
    //~^ ERROR the size for values of type `T` cannot be known
    requires_pointeesized::<T>();
}

// `T` inherits the `const MetaSized` implicit supertrait of `Bare`.
fn with_bare_trait<T: PointeeSized + Bare>() {
    requires_sized::<T>();
    //~^ ERROR the size for values of type `T` cannot be known
    requires_metasized::<T>();
    requires_pointeesized::<T>();
}

fn main() { }
