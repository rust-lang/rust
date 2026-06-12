#![feature(sized_hierarchy)]
use std::marker::{MetaSized, PointeeSized};
use std::mem::{size_of, size_of_val};
use std::fmt::Debug;

fn foo<T: only Sized>(t: T) {
    size_of::<T>();
    size_of_val(&t);
}

fn bar<T: only MetaSized>(t: T) {
    //~^ ERROR the size for values of type `T` cannot be known at compilation time
    size_of::<T>();
    size_of_val(&t);
}

fn barfoo<T: only PointeeSized>(t: T) {
    //~^ ERROR the size for values of type `T` cannot be known at compilation time
    size_of::<T>();
    size_of_val(&t);
    //~^ ERROR the size for values of type `T` cannot be known
}

trait OnlyPointeeSized: only PointeeSized {
    fn foo(&self) {
        size_of_val(self);
        //~^ ERROR the size for values of type `Self` cannot be known
    }
}

trait NonSizedOnly1: only Debug {}
//~^ ERROR `only` may only be applied to sizedness traits

trait Trait {}

trait NonSizedOnly2: only Trait {}
//~^ ERROR `only` may only be applied to sizedness traits

trait OnlyOnly: only OnlyPointeeSized {}
//~^ ERROR `only` may only be applied to sizedness traits

trait DoubleOnly: only PointeeSized + only MetaSized {}
// Redundant, but not illegal.

trait InheritedOnly: OnlyPointeeSized {
    fn foo(&self) {
        size_of_val(self);
        // This works fine, `only` is not transitive, this trait still has the
        // `MetaSized` default supertrait.
    }
}

fn main() {}
