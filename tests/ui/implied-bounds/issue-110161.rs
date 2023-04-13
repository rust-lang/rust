// ICE regression relating to unconstrained lifetimes in implied
// bounds. See #110161.

// compile-flags: --crate-type=lib

trait Trait {
    type Ty;
}

// erroneous `Ty` impl
impl Trait for () {
//~^ ERROR not all trait items implemented, missing: `Ty` [E0046]
}

// `'lt` is not constrained by the erroneous `Ty`
impl<'lt, T> Trait for Box<T>
where
    T: Trait<Ty = &'lt ()>,
{
    type Ty = &'lt ();
}

// unconstrained lifetime appears in implied bounds
fn test(_: <Box<()> as Trait>::Ty) {}
