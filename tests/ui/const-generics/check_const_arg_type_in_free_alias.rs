//@ revisions: eager lazy
#![cfg_attr(lazy, feature(lazy_type_alias))]
#![cfg_attr(lazy, expect(incomplete_features))]

// We currently do not check free type aliases for well formedness.
// However, previously we desugared all const arguments to anon
// consts which resulted in us happening to check that they were
// of the correct type. Nowadays we don't necessarily lower to a
// const argument, but to continue erroring on such code we special
// case `ConstArgHasType` clauses to be checked for free type aliases
// even though we ignore the rest of the wf requirements.

struct Foo<const N: usize>;

type ArrLen<const B: bool> = [(); B];
//~^ ERROR: the constant `B` is not of type
type AnonArrLen = [(); true];
//~^ ERROR: mismatched types
type ConstArg<const B: bool> = Foo<B>;
//~^ ERROR: the constant `B` is not of type
type AnonConstArg = Foo<true>;
//~^ ERROR: mismatched types

trait IdentityWithUnused<const N: usize> {
    type This;
}

impl<T, const N: usize> IdentityWithUnused<N> for T {
    type This = T;
}

type Alias<const B: bool> = <() as IdentityWithUnused<B>>::This;
//~^ ERROR: the constant `B` is not of type
type AnonAlias = <() as IdentityWithUnused<true>>::This;
//~^ ERROR: mismatched types

type Free<const N: usize> = [(); N];
type UseFree<const B: bool> = Free<B>;
//~^ ERROR: the constant `B` is not of type
type AnonUseFree = Free<true>;
//~^ ERROR: mismatched types

// This previously emitted an error before we stopped using
// anon consts. Now, as free aliases don't exist after ty
// lowering, we don't emit an error because we only see `N`
// being used as an argument to an array length.
//
// Free type aliases are not allowed to have unused generic
// parameters so this shouldn't be able to cause code to
// pass that should error.
type UseFreeIndirectlyCorrect<const N: usize> = UseFree<N>;
//[lazy]~^ ERROR: the constant `N` is not of type
type AnonUseFreeIndirectlyCorrect = UseFree<1_usize>;
//~^ ERROR: mismatched types

struct Wrap<T>(T);

type IndirectArr<const B: bool> = Wrap<Wrap<[(); B]>>;
//~^ ERROR: the constant `B` is not of type
type AnonIndirectArr = Wrap<Wrap<[(); true]>>;
//~^ ERROR: mismatched types
type IndirectConstArg<const B: bool> = Wrap<Wrap<Foo<B>>>;
//~^ ERROR: the constant `B` is not of type
type AnonIndirectConstArg = Wrap<Wrap<Foo<true>>>;
//~^ ERROR: mismatched types

fn main() {}
