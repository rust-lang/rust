// Weak alias types only constrain late-bound regions if their normalized form constrains them.

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type NotInjective<'a> = <() as Discard>::Output<'a>;

type FnPtr0 = for<'a> fn(NotInjective<'a>) -> &'a ();
//~^ ERROR references lifetime `'a`, which is not constrained by the fn input types
type FnPtr1 = for<'a> fn(NotInjectiveEither<'a, ()>) -> NotInjectiveEither<'a, ()>;
//~^ ERROR references lifetime `'a`, which is not constrained by the fn input types
type DynCl = dyn for<'a> Fn(NotInjective<'a>) -> &'a ();
//~^ ERROR references lifetime `'a`, which does not appear in the trait input types

trait Discard { type Output<'a>; }
impl Discard for () { type Output<'_a> = (); }

type NotInjectiveEither<'a, Linchpin> = Linchpin
where
    Linchpin: Fn() -> &'a ();


fn main() {}
