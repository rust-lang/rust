//@ run-rustfix
#![allow(dead_code)]

use std::ops::{Deref, DerefMut};

fn take(mut wrap: Wrap<[Option<NonCopy>; 1]>) {
    if let Some(mut val) = wrap[0] {
        //~^ ERROR cannot move out of type `[Option<NonCopy>; 1]`, a non-copy array
        val.0 = ();
    }
}

fn take_mut_ref_base(mut wrap: Wrap<[Option<NonCopy>; 1]>) {
    if let Some(mut val) = (&mut wrap)[0] {
        //~^ ERROR cannot move out of type `[Option<NonCopy>; 1]`, a non-copy array
        val.0 = ();
    }
}

struct Wrap<T>(T);
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
