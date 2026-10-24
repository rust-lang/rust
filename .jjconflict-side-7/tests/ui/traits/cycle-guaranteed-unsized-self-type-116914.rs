//! Regression test for <https://github.com/rust-lang/rust/issues/116914>.

trait Filter {
    type ToMatch;
}

impl<T> Filter for T //~ ERROR cycle detected when computing whether
where
    T: Fn(Self::ToMatch),
{
}

trait Rule {}

impl<T> Rule for T
where
    T: Filter,
{
}

struct JustFilter<F: Filter> {
    filter: F,
}

impl<T: Filter> Rule for JustFilter<T> {}

fn main() {}
