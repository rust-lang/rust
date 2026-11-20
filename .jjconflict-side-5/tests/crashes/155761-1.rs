//@ known-bug: #155761
//@compile-flags: -Znext-solver

// A regression test for https://github.com/rust-lang/rust/issues/151329.
// Ensures we do not trigger an ICE when normalization fails for a
// projection on a trait object, even if the projection has the same
// trait id as the object's bound.

// ICE again after moving to eager normalization in the next solver.
// #155761 has a simplified variant which causes ICE without eager normalization.

trait Foo {
    type V;
}

trait Callback<T: Foo>: Fn(&T, &T::V) {}

struct Bar<T: Foo + ?Sized> {
    callback: Box<dyn Callback<T>>,
}

impl<T: Foo> Bar<dyn Callback<T>> {
    fn event(&self) {
        (self.callback)(any(), any());
    }
}

fn any() {}

fn main() {}
