// [no_self_infer] check-pass
// compile-flags: -Ztrait-solver=next
// revisions: self_infer no_self_infer

// checks that the new solver is smart enough to infer `?0 = U` when solving:
// `normalizes-to(<Vec<?0> as Trait>::Assoc, u8)`
// with `normalizes-to(<Vec<U> as Trait>::Assoc, u8)` in the paramenv even when
// there is a separate `Vec<T>: Trait` bound  in the paramenv.
//
// FIXME(-Ztrait-solver=next)
// This could also compile for `normalizes-to(<?0 as Trait>::Assoc, u8)` but
// we currently immediately consider a goal ambiguous if the self type is an
// inference variable.

trait Trait {
    type Assoc;
}

fn foo<T: Trait<Assoc = u8>>(x: T) {}

#[cfg(self_infer)]
fn unconstrained<T>() -> T {
    todo!()
}

#[cfg(no_self_infer)]
fn unconstrained<T>() -> Vec<T> {
    todo!()
}

fn bar<T, U>()
where
    Vec<T>: Trait,
    Vec<U>: Trait<Assoc = u8>,
{
    foo(unconstrained())
    //[self_infer]~^ ERROR type annotations needed
}

fn main() {}
