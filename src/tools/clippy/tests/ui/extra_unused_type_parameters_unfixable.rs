#![warn(clippy::extra_unused_type_parameters)]

fn unused_where_clause<T, U>(x: U)
//~^ extra_unused_type_parameters
where
    T: Default,
{
    unimplemented!();
}

fn unused_multi_where_clause<T, U, V: Default>(x: U)
//~^ extra_unused_type_parameters
where
    T: Default,
{
    unimplemented!();
}

fn unused_all_where_clause<T, U: Default, V: Default>()
//~^ extra_unused_type_parameters
where
    T: Default,
{
    unimplemented!();
}

fn main() {}
