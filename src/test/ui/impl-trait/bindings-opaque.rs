#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete and may cause the compiler to crash

const FOO: impl Copy = 42;

static BAR: impl Copy = 42;

fn main() {
    let foo: impl Copy = 42;

    let _ = FOO.count_ones();
//~^ ERROR no method
    let _ = BAR.count_ones();
//~^ ERROR no method
    let _ = foo.count_ones();
//~^ ERROR no method
}
