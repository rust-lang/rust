//@ check-pass

#![feature(min_generic_const_args)]
//~^ WARN the feature `min_generic_const_args` is incomplete
#![feature(inherent_associated_types)]
//~^ WARN the feature `inherent_associated_types` is incomplete

// Test case from #138226: generic impl with multiple type parameters
struct Foo<A, B>(A, B);
impl<A, B> Foo<A, B> {
    type const LEN: usize = 4;

    fn foo() {
        let _ = [5; Self::LEN];
    }
}

// Test case from #138226: generic impl with const parameter
struct Bar<const N: usize>;
impl<const N: usize> Bar<N> {
    type const LEN: usize = 4;

    fn bar() {
        let _ = [0; Self::LEN];
    }
}

// Test case from #150960: non-generic impl with const block
struct Baz;
impl Baz {
    type const LEN: usize = 4;

    fn baz() {
        let _ = [0; { Self::LEN }];
    }
}

fn main() {}
