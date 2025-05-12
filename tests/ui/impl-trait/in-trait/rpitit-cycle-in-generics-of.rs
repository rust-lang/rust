//@ check-pass

// Check that we don't hit a query cycle when:
// 1. Computing generics_of, which requires...
// 2. Calling resolve_bound_vars, which requires...
// 3. Calling associated_items, which requires...
// 4. Calling associated_type_for_impl_trait_in_trait, which requires...
// 5. Computing generics_of, which cycles.

pub trait Foo<'a> {
    type Assoc;

    fn demo<T>(other: T) -> impl Foo<'a, Assoc = Self::Assoc>
    where
        T: Foo<'a, Assoc = ()>;
}

fn main() {}
