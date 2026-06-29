//@ compile-flags: -Znext-solver -Zassumptions-on-binders

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

fn borrowck_multiple_origins_fail<'a, 'b, T: AliasHaver, U: AliasHaver>()
where
    <T as AliasHaver>::Assoc: 'a,
    <U as AliasHaver>::Assoc: 'b,
{
    let _: ReqTrait<T::Assoc>;
    //~^ ERROR: higher-ranked lifetime bound could not be satisfied
    let _: ReqTrait<U::Assoc>;
    //~^ ERROR: higher-ranked lifetime bound could not be satisfied
}

const REGIONCK_ENV_PASS<'a, T: AliasHaver>: ReqTrait<T::Assoc> = todo!()
where
    <T as AliasHaver>::Assoc: 'static;

const REGIONCK_ENV_FAIL<'a, T: AliasHaver>: ReqTrait<T::Assoc> = todo!()
//~^ ERROR: higher-ranked lifetime bound could not be satisfied
where
    <T as AliasHaver>::Assoc: 'a;

fn main() {}
