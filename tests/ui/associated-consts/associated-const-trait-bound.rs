//@ build-pass (FIXME(62277): could be check-pass?)

trait ConstDefault {
    const DEFAULT: Self;
}

trait Foo: Sized {}

trait FooExt: Foo {
    type T: ConstDefault;
}

trait Bar<F: FooExt> {
    const T: F::T;
}

impl<F: FooExt> Bar<F> for () {
    const T: F::T = <F::T as ConstDefault>::DEFAULT;
}

fn main() {}
