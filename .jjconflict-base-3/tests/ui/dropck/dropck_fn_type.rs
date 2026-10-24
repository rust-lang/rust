//@ run-pass
//! Regression test for #58311, regarding the usage of Fn types in drop impls

// All of this Drop impls should compile.

#[allow(dead_code)]
struct S<F: Fn() -> [u8; 1]>(F);

impl<F: Fn() -> [u8; 1]> Drop for S<F> {
    fn drop(&mut self) {}
}

#[allow(dead_code)]
struct P<A, F: FnOnce() -> [A; 10]>(F);

impl<A, F: FnOnce() -> [A; 10]> Drop for P<A, F> {
    fn drop(&mut self) {}
}

fn main() {}
