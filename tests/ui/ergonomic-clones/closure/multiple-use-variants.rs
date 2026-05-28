#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::clone::UseCloned;

fn takes_val<T>(_: T) {}
fn takes_ref<'a, T>(_: &'a T) {}

#[derive(Clone)]
struct Inner<'a, T>(&'a T);

impl<'a, T> UseCloned for Inner<'a, T> where T: Clone {}

fn main() {
    let v = String::new();
    let inner = Inner(&v);

    let _ = use || {
        takes_ref(inner.0);
        takes_val(inner.0)
    };
    let _ = use || {
        takes_ref(inner.0);
        takes_val(inner.0);
        takes_val(inner.0);
        takes_val(inner)
    };
    let _ = use || {
        takes_ref(inner.0);
        takes_val(inner.0);
        takes_val(inner);
        takes_val(inner)
        //~^ ERROR: use of moved value: `inner` [E0382]
    };
}
