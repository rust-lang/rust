//@ compile-flags: -Zassumptions-on-binders

// test that a `<T as AliasHaver>::Assoc: '!a_u1` constraint is considered to be satisfied
// if there's a `T::Assoc: 'static` assumption in the root universe and if not that it is
// an error :)

#![feature(generic_const_items)]

trait AliasHaver {
    type Assoc;
}

trait Trait<'a> {}
impl<'a, T: 'a> Trait<'a> for T {}

struct ReqTrait<T: for<'a> Trait<'a>>(T);

fn borrowck_env_pass<'a, T: AliasHaver>()
where
    <T as AliasHaver>::Assoc: 'static,
{
    let _: ReqTrait<T::Assoc>;
}

fn borrowck_env_fail<'a, T: AliasHaver>()
where
    <T as AliasHaver>::Assoc: 'a,
{
    let _: ReqTrait<T::Assoc>;
    //~^ ERROR: higher-ranked lifetime bound could not be satisfied
}

const REGIONCK_ENV_PASS<'a, T: AliasHaver>: ReqTrait<T::Assoc> = todo!()
where
    <T as AliasHaver>::Assoc: 'static;

const REGIONCK_ENV_FAIL<'a, T: AliasHaver>: ReqTrait<T::Assoc> = todo!()
//~^ ERROR: higher-ranked lifetime bound could not be satisfied
where
    <T as AliasHaver>::Assoc: 'a;

// Solver constraints produced while normalizing implied bounds must be returned
// to lexical regionck.
trait Project {
    type Assoc;
}

impl<T: AliasHaver> Project for (T,)
where
    T::Assoc: for<'a> Trait<'a>,
{
    type Assoc = ();
}

struct Normalizes<T: Project>(T)
where
    T::Assoc: Clone;

trait TestTrait {}

impl<'a, T: AliasHaver> TestTrait for [Normalizes<(T,)>; 1]
//~^ ERROR: higher-ranked lifetime bound could not be satisfied
where
    T::Assoc: 'a,
{
}

fn main() {}
