//@ compile-flags: -Znext-solver
//@ check-pass
#![allow(unconditional_recursion)]

trait Bound<T> {}

trait Trait<T> {}

impl<T, U> Trait<U> for T
where
    T: Bound<U>,
{
}

struct Output {
    field: u32,
}

struct Hidden;

impl Bound<Output> for Hidden {}

fn use_trait<T: Trait<U>, U>(_: T) -> U {
    todo!()
}

fn opaque() -> impl Bound<Output> {
    let output = use_trait(opaque());

    // The blanket impl itself must not constrain `U`, but its `T: Bound<U>`
    // requirement may use the opaque's item bounds to guide inference.
    output.field;

    Hidden
}

fn main() {}
