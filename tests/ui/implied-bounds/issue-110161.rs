// ICE regression relating to unconstrained lifetimes in implied
// bounds. See #110161.

//@ compile-flags: --crate-type=lib

trait LtTrait {
    type Ty;
}

// erroneous `Ty` impl
impl LtTrait for () {
//~^ ERROR not all trait items implemented, missing: `Ty` [E0046]
}

// `'lt` is not constrained by the erroneous `Ty`
impl<'lt, T> LtTrait for Box<T>
where
    T: LtTrait<Ty = &'lt ()>,
{
    type Ty = &'lt ();
}

// unconstrained lifetime appears in implied bounds
fn test(_: <Box<()> as LtTrait>::Ty) {}

fn test2<'x>(_: &'x <Box<()> as LtTrait>::Ty) {}
