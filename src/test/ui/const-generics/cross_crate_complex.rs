// aux-build:crayte.rs
// edition:2018
// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]
extern crate crayte;

use crayte::*;

async fn foo() {
    in_foo(out_foo::<3>());
    async_simple([0; 17]).await;
    async_in_foo(async_out_foo::<4>().await).await;
}

struct Faz<const N: usize>;

impl<const N: usize> Foo<N> for Faz<N> {}
impl<const N: usize> Bar<N> for Faz<N> {
    type Assoc = Faz<N>;
}

fn main() {
    let _ = foo;
}
