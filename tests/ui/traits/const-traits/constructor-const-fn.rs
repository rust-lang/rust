//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]
const fn impls_fn<F: [const] Fn(u32) -> Foo>(_: &F) {}

struct Foo(u32);

const fn foo() {
    // This previously triggered an incorrect assert
    // when checking whether the constructor of `Foo`
    // is const.
    impls_fn(&Foo)
}

fn main() {}
