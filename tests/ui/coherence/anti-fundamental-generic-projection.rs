//@ aux-build: anti_fundamental_trait_lib.rs

// Test that generic projections cannot bypass #[rustc_anti_fundamental]
// due to normalization failure or compat mode.

extern crate anti_fundamental_trait_lib;

use anti_fundamental_trait_lib::{
    AntiFundamentalTrait, AntiFundamentalWithParam, FundamentalWrapper,
};

struct LocalGeneric<T>(T);
struct LocalGeneric2<T>(T);

trait GenericAssocHelper<T> {
    type Assoc;
}

impl<T> GenericAssocHelper<T> for LocalGeneric<T> {
    type Assoc = FundamentalWrapper<LocalGeneric<T>>;
}

// ERROR: Generic projections normalize at coherence time, but
// we reject as a hard error because Compat mode is disallowed for anti-fundamental traits.
impl<T> AntiFundamentalWithParam<LocalGeneric<T>>
//~^ ERROR type parameter `T` must be covered by another type
    for <LocalGeneric<T> as GenericAssocHelper<T>>::Assoc
{
}

// ERROR: Generic projection with bounds is similarly hard rejected.
trait BoundedHelper<T> {
    type Assoc;
}

impl<T: Clone> BoundedHelper<T> for LocalGeneric2<T> {
    type Assoc = FundamentalWrapper<LocalGeneric2<T>>;
}

impl<T: Clone> AntiFundamentalWithParam<LocalGeneric2<T>>
//~^ ERROR type parameter `T` must be covered by another type
    for <LocalGeneric2<T> as BoundedHelper<T>>::Assoc
{
}

// Blanket implementation attempting to implement an anti-fundamental trait
// on a projection where the type parameter is constrained by the trait:
// rejected by the orphan rule (E0210) because T is not behind a local type.
trait Helper2 {
    type T;
}

impl<T: Helper2> AntiFundamentalWithParam<T> for <T as Helper2>::T {
    //~^ ERROR type parameter `T` must be used as an argument to some local type
}

impl<T> Helper2 for FundamentalWrapper<T> {
    type T = T;
}

// Blanket implementation of a parameterless anti-fundamental trait (like `Deref`)
// on an associated type projection: rejected both by the orphan rule (E0210)
// and because T is unconstrained (E0207).
trait Helper3 {
    type T;
}

impl<T: Helper3> AntiFundamentalTrait for <T as Helper3>::T {
    //~^ ERROR type parameter `T` must be used as an argument to some local type
    //~| ERROR the type parameter `T` is not constrained
}

impl<T> Helper3 for FundamentalWrapper<T> {
    type T = T;
}

fn main() {}
