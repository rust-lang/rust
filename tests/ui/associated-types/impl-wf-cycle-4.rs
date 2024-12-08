trait Filter {
    type ToMatch;
}

impl<T> Filter for T //~ ERROR overflow evaluating the requirement
where
    T: Fn(Self::ToMatch),
{
}

struct JustFilter<F: Filter> {
    filter: F,
}

fn main() {}
