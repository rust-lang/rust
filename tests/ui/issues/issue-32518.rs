//@ run-pass
//@ no-prefer-dynamic
//@ aux-build:cgu_test.rs
//@ aux-build:cgu_test_a.rs
//@ aux-build:cgu_test_b.rs

extern crate cgu_test_a;
extern crate cgu_test_b;

fn main() {
    cgu_test_a::a::a();
    cgu_test_b::a::a();
}
