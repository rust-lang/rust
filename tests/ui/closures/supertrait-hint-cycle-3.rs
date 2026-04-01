//@ check-pass


trait Foo<'a> {
    type Input;
}

impl<F: Fn(u32)> Foo<'_> for F {
    type Input = u32;
}

fn needs_super<F: for<'a> Fn(<F as Foo<'a>>::Input) + for<'a> Foo<'a>>(_: F) {}

fn main() {
    needs_super(|_: u32| {});
}
