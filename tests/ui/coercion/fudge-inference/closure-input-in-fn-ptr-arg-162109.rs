//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// A regression test for https://github.com/rust-lang/rust/issues/162109.

struct Foo<I, O>(I, fn(I) -> O);

impl<I, O> Foo<I, O> {
    fn new(_: I, _: fn(I) -> O) -> Self {
        loop {}
    }
}

fn main() {
    let _: Foo<_, u8> = Foo::new(&1u8, |input| *input);
}
