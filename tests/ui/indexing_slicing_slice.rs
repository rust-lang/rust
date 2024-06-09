//@aux-build: proc_macros.rs

#![warn(clippy::indexing_slicing)]
// We also check the out_of_bounds_indexing lint here, because it lints similar things and
// we want to avoid false positives.
#![warn(clippy::out_of_bounds_indexing)]
#![allow(
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::useless_vec,
    unused_must_use,
    unused
)]
#![warn(clippy::indexing_slicing)]

extern crate proc_macros;
use proc_macros::with_span;

use std::ops::Index;

struct BoolMap<T> {
    false_value: T,
    true_value: T,
}

impl<T> Index<bool> for BoolMap<T> {
    type Output = T;
    fn index(&self, index: bool) -> &T {
        if index { &self.true_value } else { &self.false_value }
    }
}

struct BoolMapWithGet<T> {
    false_value: T,
    true_value: T,
}

impl<T> Index<bool> for BoolMapWithGet<T> {
    type Output = T;
    fn index(&self, index: bool) -> &Self::Output {
        if index { &self.true_value } else { &self.false_value }
    }
}

impl<T> BoolMapWithGet<T> {
    fn get(&self, index: bool) -> Option<&T> {
        if index {
            Some(&self.true_value)
        } else {
            Some(&self.false_value)
        }
    }
}

struct S<T>(T);
impl S<i32> {
    fn get() -> Option<i32> {
        unimplemented!()
    }
}
impl<T> Index<i32> for S<T> {
    type Output = T;
    fn index(&self, _index: i32) -> &Self::Output {
        &self.0
    }
}

struct Y<T>(T);
impl Y<i32> {
    fn get<U>() -> Option<U> {
        unimplemented!()
    }
}
impl<T> Index<i32> for Y<T> {
    type Output = T;
    fn index(&self, _index: i32) -> &Self::Output {
        &self.0
    }
}

struct Z<T>(T);
impl<T> Z<T> {
    fn get<T2>() -> T2 {
        unimplemented!()
    }
}
impl<T> Index<i32> for Z<T> {
    type Output = T;
    fn index(&self, _index: i32) -> &Self::Output {
        &self.0
    }
}

with_span!(
    span

    fn dont_lint_proc_macro() {
        let x = [1, 2, 3, 4];
        let index: usize = 1;
        &x[index..];
        &x[..10];

        let x = vec![0; 5];
        let index: usize = 1;
        &x[index..];
        &x[..10];
    }
);

fn main() {
    let x = [1, 2, 3, 4];
    let index: usize = 1;
    let index_from: usize = 2;
    let index_to: usize = 3;
    &x[index..];
    //~^ ERROR: slicing may panic
    &x[..index];
    //~^ ERROR: slicing may panic
    &x[index_from..index_to];
    //~^ ERROR: slicing may panic
    &x[index_from..][..index_to];
    //~^ ERROR: slicing may panic
    //~| ERROR: slicing may panic
    &x[5..][..10];
    //~^ ERROR: slicing may panic
    //~| ERROR: range is out of bounds
    //~| NOTE: `-D clippy::out-of-bounds-indexing` implied by `-D warnings`
    &x[0..][..3];
    //~^ ERROR: slicing may panic
    &x[1..][..5];
    //~^ ERROR: slicing may panic

    &x[0..].get(..3); // Ok, should not produce stderr.
    &x[0..3]; // Ok, should not produce stderr.

    let y = &x;
    &y[1..2];
    &y[0..=4];
    //~^ ERROR: range is out of bounds
    &y[..=4];
    //~^ ERROR: range is out of bounds

    &y[..]; // Ok, should not produce stderr.

    let v = vec![0; 5];
    &v[10..100];
    //~^ ERROR: slicing may panic
    &x[10..][..100];
    //~^ ERROR: slicing may panic
    //~| ERROR: range is out of bounds
    &v[10..];
    //~^ ERROR: slicing may panic
    &v[..100];
    //~^ ERROR: slicing may panic

    &v[..]; // Ok, should not produce stderr.

    let map = BoolMap {
        false_value: 2,
        true_value: 4,
    };

    map[true]; // Ok, because `get` does not exist (custom indexing)

    let map_with_get = BoolMapWithGet {
        false_value: 2,
        true_value: 4,
    };

    // Lint on this, because `get` does exist with same signature
    map_with_get[true];

    let s = S::<i32>(1);
    s[0];

    let y = Y::<i32>(1);
    y[0];

    let z = Z::<i32>(1);
    z[0];
}
