//@no-rustfix

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

// The fix just removes the type parameter from the definition of `unused_ty`, but it doesn't adjust
// its callsites, leading to compilation errors.
mod issue15884 {
    fn unused_ty<T>(x: u8) {
        //~^ extra_unused_type_parameters
        unimplemented!()
    }

    fn main() {
        unused_ty::<String>(0);
    }
}

fn main() {}
