//@ check-pass

#![feature(gca_min_const_items, gca_macroless_args, inherent_associated_types)]

use std::gca;

// Test case from #138226: generic impl with multiple type parameters
struct Foo<A, B>(A, B);
impl<A, B> Foo<A, B> {
    const LEN: usize = gca!(4);

    fn foo() {
        let _ = [5; Self::LEN];
    }
}

// Test case from #138226: generic impl with const parameter
struct Bar<const N: usize>;
impl<const N: usize> Bar<N> {
    const LEN: usize = gca!(4);

    fn bar() {
        let _ = [0; Self::LEN];
    }
}

// Test case from #150960: non-generic impl with const block
struct Baz;
impl Baz {
    const LEN: usize = gca!(4);

    fn baz() {
        let _ = [0; { Self::LEN }];
    }
}

fn main() {}
