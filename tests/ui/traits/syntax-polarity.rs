//@ run-pass
#![allow(dead_code)]

#![feature(negative_impls)]

struct TestType;

impl TestType {}

trait TestTrait {}

impl !Send for TestType {}

struct TestType2<T>(T);

impl<T> TestType2<T> {}

impl<T> !Send for TestType2<T> {}

fn main() {}
