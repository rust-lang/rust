// Regression test for #132767.
// This used to ICE with "maybe try to call `try_normalize_erasing_regions`"
// when normalizing the hidden type of `impl Trait`.

trait Func {
    type Ret;
}

impl<F: FnOnce() -> R, R> Func for F {
    type Ret = R;
}

trait Id {}

fn invalid_impl_trait() -> impl Id {}
//~^ ERROR the trait bound `(): Id` is not satisfied

struct Foo<T: Func> {
    _func: T,
    value: Option<<T as Func>::Ret>,
}

fn main() {
    let foo = Foo {
        _func: invalid_impl_trait,
        value: None,
    };
    drop(foo.value);
}
