#![feature(discriminant_kind)]

use std::marker::DiscriminantKind;

enum Uninhabited {}

struct NewType;

impl DiscriminantKind for NewType {
    //~^ ERROR explicit impls for the `DiscriminantKind` trait are not permitted
    type Discriminant = Uninhabited;
}

fn main() {}
