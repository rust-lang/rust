fn foo<F: Foo<N=3>>() {}
const TEST: usize = 3;
fn bar<F: Foo<N={TEST}>>() {}
