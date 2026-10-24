//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![feature(field_projections, freeze)]
#![expect(incomplete_features, dead_code)]

use std::field::field_of;

struct Struct {
    field: u32,
}

type Alias = Struct;

trait Trait {
    type Assoc;
}

impl Trait for Struct {
    type Assoc = Struct;
}

fn main() {
    let _: field_of!(Alias, field);
    let _: field_of!(<Struct as Trait>::Assoc, field);
    //~^ ERROR: could not resolve fields of `<Struct as Trait>::Assoc`
}

trait Special {
    type Assoc;
}

trait Constraint: Trait<Assoc = Struct> {}

impl Special for Struct {
    type Assoc = Self;
}

fn with_constraint1<T: Constraint>(
    _: field_of!(<<T as Trait>::Assoc as Special>::Assoc, field),
    //~^ ERROR: could not resolve fields of `<<T as Trait>::Assoc as Special>::Assoc`
) {
}

fn with_constraint2<T: Constraint>(
    _x: field_of!(<<T as Trait>::Assoc as Special>::Assoc, field),
    //~^ ERROR: could not resolve fields of `<<T as Trait>::Assoc as Special>::Assoc`
) {
}

fn with_constraint3<T: Constraint>() {
    let _: field_of!(<<T as Trait>::Assoc as Special>::Assoc, field);
    //~^ ERROR: could not resolve fields of `<<T as Trait>::Assoc as Special>::Assoc`
}

fn with_constraint_invalid_field<T: Constraint>() {
    let _: field_of!(<<T as Trait>::Assoc as Special>::Assoc, other);
    //~^ ERROR: could not resolve fields of `<<T as Trait>::Assoc as Special>::Assoc`
}
