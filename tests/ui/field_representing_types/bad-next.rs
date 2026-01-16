//@ compile-flags: -Znext-solver
#![feature(field_projections, freeze)]
#![expect(incomplete_features, dead_code)]
use std::field::field_of;

struct Struct {
    field: i32,
}

trait WithAssoc {
    type Assoc: Special;
}

trait Special {
    type Assoc;
}

trait Constraint: WithAssoc<Assoc = Struct> {}

impl Special for Struct {
    type Assoc = Self;
}

fn works<T: Constraint>(_: field_of!(<<T as WithAssoc>::Assoc as Special>::Assoc, field)) {}

// FIXME(FRTs): correct the error printing for infer vars
fn fails1<T: Constraint>(_x: field_of!(<<T as WithAssoc>::Assoc as Special>::Assoc, field)) {}
//~^ ERROR: cannot use `_` in this position

fn fails2<T: Constraint>() {
    let _: field_of!(<<T as WithAssoc>::Assoc as Special>::Assoc, field);
    // last time I checked this, this line above produced an ICE due to a `delay_bug`
}

fn invalid_field<T: Constraint>() {
    // FIXME(FRTs): don't report the error multiple times?
    let _: field_of!(<<T as WithAssoc>::Assoc as Special>::Assoc, other);
    //~^ ERROR: no field `other` on struct `Struct`
    //~^^ ERROR: no field `other` on struct `Struct`
    //~^^^ ERROR: no field `other` on struct `Struct`
}

fn main() {}
