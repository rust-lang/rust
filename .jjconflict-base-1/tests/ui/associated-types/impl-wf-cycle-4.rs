trait Filter {
    type ToMatch;
}

impl<T> Filter for T //~ ERROR cycle detected when
where
    T: Fn(Self::ToMatch),
{
}

struct JustFilter<F: Filter> {
    filter: F,
}

fn main() {}
