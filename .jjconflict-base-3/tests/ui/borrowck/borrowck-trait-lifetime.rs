//@ run-pass
#![allow(unused_imports)]
// This test verifies that casting from the same lifetime on a value
// to the same lifetime on a trait succeeds. See issue #10766.


#![allow(dead_code)]

use std::marker;

fn main() {
    trait T { fn foo(&self) {} }

    fn f<'a, V: T>(v: &'a V) -> &'a dyn T {
        v as &'a dyn T
    }
}
