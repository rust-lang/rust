// compile-flags: -Ztrait-solver=next
// check-pass

// Checks that we do not get ambiguity by considering an impl
// multiple times if we're able to normalize the self type.

trait Trait<'a> {}

impl<'a, T: 'a> Trait<'a> for T {}

fn impls_trait<'a, T: Trait<'a>>() {}

trait Id {
    type Assoc;
}
impl<T> Id for T {
    type Assoc = T;
}

fn call<T>() {
    impls_trait::<<T as Id>::Assoc>();
}

fn main() {
    call::<()>();
    impls_trait::<<<() as Id>::Assoc as Id>::Assoc>();
}
