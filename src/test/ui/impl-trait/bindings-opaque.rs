#![feature(impl_trait_in_bindings)]

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
