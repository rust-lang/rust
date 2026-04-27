//@ run-rustfix
#![allow(dead_code)]

use std::ops::{Deref, DerefMut};

fn take(mut wrap: Wrap<Struct>) {
    if let Some(mut val) = wrap.field {
        //~^ ERROR cannot move out of dereference of `Wrap<Struct>`
        val.0 = ();
    }
}

fn take_mut_ref_base(mut wrap: Wrap<Struct>) {
    if let Some(mut val) = (&mut wrap).field {
        //~^ ERROR cannot move out of dereference of `Wrap<Struct>`
        val.0 = ();
    }
}

struct Wrap<T>(T);
struct Struct {
    field: Option<NonCopy>,
}
struct NonCopy(());

impl<T> Deref for Wrap<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Wrap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn main() {}
