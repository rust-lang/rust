#![allow(incomplete_features)]
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
