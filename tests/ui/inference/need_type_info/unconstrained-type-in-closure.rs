//! Regression test for <https://github.com/rust-lang/rust/issues/17551>.

use std::marker;

struct B<T>(marker::PhantomData<T>);

fn main() {
    let foo = B(marker::PhantomData); //~ ERROR type annotations needed
    let closure = || foo;
}
