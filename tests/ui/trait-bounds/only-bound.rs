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

fn main() {}
