// check-pass
// known-bug: #97156

#![feature(const_type_id, generic_const_exprs)]
#![allow(incomplete_features)]

use std::any::TypeId;
// `One` and `Two` are currently considered equal types, as both
// `One <: Two` and `One :> Two` holds.
type One = for<'a> fn(&'a (), &'a ());
type Two = for<'a, 'b> fn(&'a (), &'b ());
trait AssocCt {
    const ASSOC: usize;
}
const fn to_usize<T: 'static>() -> usize {
    const WHAT_A_TYPE: TypeId = TypeId::of::<One>();
    match TypeId::of::<T>() {
        WHAT_A_TYPE => 0,
        _ => 1000,
    } 
}
impl<T: 'static> AssocCt for T {
    const ASSOC: usize = to_usize::<T>();
}

trait WithAssoc<U> {
    type Assoc;
}
impl<T: 'static> WithAssoc<()> for T where [(); <T as AssocCt>::ASSOC]: {
    type Assoc = [u8; <T as AssocCt>::ASSOC];
}

fn generic<T: 'static, U>(x: <T as WithAssoc<U>>::Assoc) -> <T as WithAssoc<U>>::Assoc
where
    [(); <T as AssocCt>::ASSOC]:,
    T: WithAssoc<U>,
{
    x
}


fn unsound<T>(x: <One as WithAssoc<T>>::Assoc) -> <Two as WithAssoc<T>>::Assoc
where
    One: WithAssoc<T>,
{
    let x: <Two as WithAssoc<T>>::Assoc = generic::<One, T>(x);
    x
}

fn main() {
    println!("{:?}", unsound::<()>([]));
}
