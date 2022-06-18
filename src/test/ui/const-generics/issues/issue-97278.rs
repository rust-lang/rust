#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::sync::Arc;

#[derive(PartialEq, Eq)]
enum Bar {
    Bar(Arc<i32>)
}

fn test<const BAR: Bar>() {}
//~^ ERROR `Arc<i32>` must be annotated with `#[derive(PartialEq, Eq)]`

fn main() {}
