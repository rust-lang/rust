//@ compile-flags: -Znext-solver

// Checks whether the new solver is smart enough to infer `?0 = U` when solving:
// `normalizes-to(<Vec<?0> as Trait>::Assoc, u8)`
// with `normalizes-to(<Vec<U> as Trait>::Assoc, u8)` in the paramenv even when
// there is a separate `Vec<T>: Trait` bound  in the paramenv.
//
// We currently intentionally do not guide inference this way.

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
    //~^ ERROR type annotations needed
}

fn main() {}
