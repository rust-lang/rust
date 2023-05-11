#![warn(clippy::extra_unused_type_parameters)]

fn unused_where_clause<T, U>(x: U)
where
    T: Default,
{
    unimplemented!();
}

fn unused_multi_where_clause<T, U, V: Default>(x: U)
where
    T: Default,
{
    unimplemented!();
}

fn unused_all_where_clause<T, U: Default, V: Default>()
where
    T: Default,
{
    unimplemented!();
}

fn main() {}
