//@ check-pass
// https://github.com/rust-lang/rust/issues/129541

#[derive(Clone)]
struct Test {
    field: std::borrow::Cow<'static, [Self]>,
}

#[derive(Clone)]
struct Hello {
    a: <[Hello] as std::borrow::ToOwned>::Owned,
}

fn main(){}
