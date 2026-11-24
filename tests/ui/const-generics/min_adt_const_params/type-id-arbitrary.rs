//@ edition:2024
#![feature(min_adt_const_params, const_param_ty_trait, type_id_structural_match)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;
use std::any::TypeId;

#[derive(PartialEq, Eq, ConstParamTy)]
struct Value {
    disambiguator: TypeId,
}

macro_rules! arbitrary_const_value {
    () => {
        const {
            enum Arbitrary {}
            Value { disambiguator: TypeId::of::<Arbitrary>() }
        }
    }
}

struct Wrapper<const V: Value>;

fn main() {
    let a: Wrapper<{ arbitrary_const_value!() }> = Wrapper;
    let b: Wrapper<{ arbitrary_const_value!() }> = a;
    //~^ ERROR mismatched types
}
