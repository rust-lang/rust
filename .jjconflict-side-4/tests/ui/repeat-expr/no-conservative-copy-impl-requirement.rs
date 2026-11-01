struct Foo<const N: usize>;

impl Clone for Foo<1> {
    fn clone(&self) -> Self {
        Foo
    }
}
impl Copy for Foo<1> {}

fn unify<const N: usize>(_: &[Foo<N>; N]) {
    loop {}
}

fn main() {
    let x = &[Foo::<_>; _];
    //~^ ERROR: type annotations needed for `&[Foo<_>; _]`
    _ = unify(x);
}
