// build-pass
// edition:2021
// compile-flags: -Cdebuginfo=2

// We were not normalizing opaques with escaping bound vars during codegen,
// leading to later linker errors because of differences in mangled symbol name.

fn func<T>() -> impl Sized {}

trait Trait<'a> {
    type Assoc;

    fn call() {
        let _ = async {
            let _value = func::<Self::Assoc>();
            std::future::ready(()).await
        };
    }
}

impl Trait<'static> for () {
    type Assoc = ();
}

fn main() {
    <()>::call();
}
