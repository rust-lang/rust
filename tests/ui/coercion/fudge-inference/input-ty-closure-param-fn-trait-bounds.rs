//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/153816>

struct Inv<T, U>(*mut (T, U));

fn pass_through<F>(_: F) -> Inv<F, F> {
    todo!()
}

fn map(_: Inv<impl FnOnce(), impl Fn()>) {}

fn traverse() {
    map(pass_through(|| ()))
}

fn main() {}
