fn a() -> Foo<bar::Baz> {}

fn b(_: impl FnMut(x::Y)) {}

fn c(_: impl FnMut(&x::Y)) {}
