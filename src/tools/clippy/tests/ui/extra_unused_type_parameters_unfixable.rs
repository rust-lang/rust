#![warn(clippy::extra_unused_type_parameters)]

fn unused_where_clause<T, U>(x: U)
//~^ ERROR: type parameter `T` goes unused in function definition
where
    T: Default,
{
    unimplemented!();
}

fn unused_multi_where_clause<T, U, V: Default>(x: U)
//~^ ERROR: type parameters go unused in function definition: T, V
where
    T: Default,
{
    unimplemented!();
}

fn unused_all_where_clause<T, U: Default, V: Default>()
//~^ ERROR: type parameters go unused in function definition: T, U, V
where
    T: Default,
{
    unimplemented!();
}

fn main() {}
