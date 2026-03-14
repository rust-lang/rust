//@ check-pass
//@ compile-flags: -Znext-solver

// Checks whether the new solver is smart enough to infer `?0 = U` when solving:
// `normalizes-to(<Vec<?0> as Trait>::Assoc, u8)`
// with `normalizes-to(<Vec<U> as Trait>::Assoc, u8)` in the paramenv even when
// there is a separate `Vec<T>: Trait` bound  in the paramenv.
//
// Since we skip proving the trait goal in normalizes-to goal now, the normalizes-to
// goal can successfully resolve the infer var via param env.
// This causes the stalled ambiguous trait goal to succeed as well.

trait Trait {
    type Assoc;
}

fn foo<T: Trait<Assoc = u8>>(x: T) {}

fn unconstrained<T>() -> Vec<T> {
    todo!()
}

fn bar<T, U>()
where
    Vec<T>: Trait,
    Vec<U>: Trait<Assoc = u8>,
{
    foo(unconstrained())
}

fn main() {}
