//@ compile-flags: -Znext-solver
//@ check-pass
#![allow(unconditional_recursion)]

trait Trait {
    type Assoc;
}

trait OtherTrait<U> {}

impl<T, U> OtherTrait<U> for T
where
    T: Trait<Assoc = U>,
{
}

struct Output {
    field: u32,
}

struct Hidden;

impl Trait for Hidden {
    type Assoc = Output;
}

fn use_other<T: OtherTrait<U>, U>(_: T) -> U {
    todo!()
}

fn opaque() -> impl Trait<Assoc = Output> {
    let output = use_other(opaque());

    // `U` is inferred from the opaque's associated-type bound rather than
    // from matching the blanket impl header.
    output.field;

    Hidden
}

fn main() {}
