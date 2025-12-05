// Make sure extern types are !Sync and !Send.

#![feature(extern_types, sized_hierarchy)]

use std::marker::PointeeSized;

extern "C" {
    type A;
}

fn assert_sync<T: PointeeSized + Sync>() {}
fn assert_send<T: PointeeSized + Send>() {}

fn main() {
    assert_sync::<A>();
    //~^ ERROR `A` cannot be shared between threads safely [E0277]

    assert_send::<A>();
    //~^ ERROR `A` cannot be sent between threads safely [E0277]
}
