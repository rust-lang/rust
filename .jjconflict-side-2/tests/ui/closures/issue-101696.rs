//@ check-pass

use std::marker::PhantomData;

#[derive(Default)]
struct MyType<'a> {
    field: usize,
    _phantom: PhantomData<&'a ()>,
}

#[derive(Default)]
struct MyTypeVariant<'a> {
    field: usize,
    _phantom: PhantomData<&'a ()>,
}

trait AsVariantTrait {
    type Type;
}

impl<'a> AsVariantTrait for MyType<'a> {
    type Type = MyTypeVariant<'a>;
}

type Variant<G> = <G as AsVariantTrait>::Type;

fn foo<T: Default, F: FnOnce(T)>(f: F) {
    let input = T::default();
    f(input);
}

fn main() {
    foo(|a: <MyType as AsVariantTrait>::Type| {
        a.field;
    });
}
