//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/153816>

pub trait Trait {
    type Assoc<F1, F2>;
    fn create_from<F>(_: F) -> Self::Assoc<F, F>;
}

fn map<T: Trait>(_: T::Assoc<impl FnOnce(), impl Fn()>) {}

pub fn traverse<T: Trait>() {
    map::<T>(T::create_from(|| ()));
}

fn main() {}
