//@ run-pass

// Makes sure we don't propagate generic instances of `Self: ?Sized` blanket impls.
// This is relevant when we have an overlapping impl and builtin dyn instance.
// See <https://github.com/rust-lang/rust/pull/114941> for more context.

trait Trait {
    fn foo(&self) -> &'static str;
}

impl<T: ?Sized> Trait for T {
    fn foo(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

fn bar<T: ?Sized>() -> fn(&T) -> &'static str {
    const { Trait::foo as fn(&T) -> &'static str }
    // If const prop were to propagate the instance
}

fn main() {
    assert_eq!("i32", bar::<dyn Trait>()(&1i32));
}
