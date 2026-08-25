//@ check-pass
// Regression test for issue https://github.com/rust-lang/rust/issues/143821
// Tests that we don't ICE when borrow-checking nested closures with generic type parameters
// and late-bound lifetime parameters.

fn data_<T: 'static>(_: &()) -> &T {
    loop {}
}

fn register<T, F>(f: F) -> IfaceToken<T>
where
    T: 'static,
    F: FnOnce(&()),
{
    loop {}
}

fn method_with_cr_async<CB>(cb: CB)
where
    CB: Fn(),
{
    loop {}
}

struct IfaceToken<T: 'static>(T);

fn foo<T>() -> IfaceToken<T> {
    register::<T, _>(|b: &()| {
        method_with_cr_async(|| {
            data_::<T>(&());
        });
    })
}

struct A();

fn main() {
    foo::<A>();
}
