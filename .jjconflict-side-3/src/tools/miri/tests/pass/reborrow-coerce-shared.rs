// Regression test for the Miri reproducer in rust-lang/rust#156313.
//
// The issue's exact recursive example ICEd while evaluating the argument
// conversion before the recursion mattered. This keeps that same
// `CustomMut`-to-`CustomRef` call path, but terminates after one recursive
// call so it can be a pass test once the ICE is fixed.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

#[allow(unused)]
struct CustomMut<'a, T>(&'a mut T);
impl<'a, T> Reborrow for CustomMut<'a, T> {}
impl<'a, T> CoerceShared<CustomRef<'a, T>> for CustomMut<'a, T> {}

struct CustomRef<'a, T>(&'a T);

impl<'a, T> Clone for CustomRef<'a, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T> Copy for CustomRef<'a, T> {}

fn method(_a: CustomRef<'_, ()>) {}

fn recursive_method(_a: CustomRef<'_, ()>, recurse: bool) {
    if recurse {
        let a = CustomMut(&mut ());
        recursive_method(a, false);
    }
}

fn issue_156313_runtime_reproducer() {
    let a = CustomMut(&mut ());
    method(a);
}

fn issue_156313_recursive_call_reproducer() {
    let a = CustomMut(&mut ());
    recursive_method(a, true);
}

fn main() {
    issue_156313_runtime_reproducer();
    issue_156313_recursive_call_reproducer();
}
