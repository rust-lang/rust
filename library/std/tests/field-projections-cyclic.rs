//@ revisions: old next
//@ [next]: compile-flags: -Znext-solver
#![allow(incomplete_features, dead_code)]
#![feature(field_projections)]

use std::field::{Field, field_of};
use std::marker::PhantomData;

struct Work<F: Field> {
    _phantom: PhantomData<F>,
}

struct MyMultiWork {
    a: Work<field_of!(Self, a)>,
    b: Work<field_of!(Self, b)>,
}

fn main() {}
