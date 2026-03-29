//@ check-fail
#![feature(sized_hierarchy)]

use std::marker::{SizeOfVal, PointeeSized};

trait Sized_: Sized { }

trait NegSized: ?Sized { }
//~^ ERROR relaxed bounds are not permitted in supertrait bounds

trait SizeOfVal_: SizeOfVal { }

trait NegSizeOfVal: ?SizeOfVal { }
//~^ ERROR relaxed bounds are not permitted in supertrait bounds
//~| ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`

trait PointeeSized_: PointeeSized { }

trait NegPointeeSized: ?PointeeSized { }
//~^ ERROR relaxed bounds are not permitted in supertrait bounds
//~| ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`

trait Bare {}

fn requires_sized<T: Sized>() {}
fn requires_sizeofval<T: SizeOfVal>() {}
fn requires_pointeesized<T: PointeeSized>() {}

fn with_sized_supertrait<T: PointeeSized + Sized_>() {
    requires_sized::<T>();
    requires_sizeofval::<T>();
    requires_pointeesized::<T>();
}

fn with_sizeofval_supertrait<T: PointeeSized + SizeOfVal_>() {
    requires_sized::<T>();
    //~^ ERROR the size for values of type `T` cannot be known at compilation time
    requires_sizeofval::<T>();
    requires_pointeesized::<T>();
}

// It isn't really possible to write this one..
fn with_pointeesized_supertrait<T: PointeeSized + PointeeSized_>() {
    requires_sized::<T>();
    //~^ ERROR the size for values of type `T` cannot be known
    requires_sizeofval::<T>();
    //~^ ERROR the size for values of type `T` cannot be known
    requires_pointeesized::<T>();
}

// `T` inherits the `const SizeOfVal` implicit supertrait of `Bare`.
fn with_bare_trait<T: PointeeSized + Bare>() {
    requires_sized::<T>();
    //~^ ERROR the size for values of type `T` cannot be known
    requires_sizeofval::<T>();
    requires_pointeesized::<T>();
}

fn main() { }
