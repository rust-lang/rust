//@ revisions: e2015 e2018 e2021
//@ [e2015] edition: 2015
//@ [e2018] edition: 2018
//@ [e2021] edition: 2021..

pub mod x {
    pub struct Struct;
    pub enum Enum {}
    pub trait Trait {}

    pub mod y {
        pub mod z {}

        type H = super::Struct::self; //~ ERROR: ambiguous associated type
        type J = super::Trait::self;
        //[e2015]~^ WARN: trait objects without an explicit `dyn` are deprecated
        //[e2015]~^^ WARN: this is accepted in the current edition
        //[e2018]~^^^ WARN: trait objects without an explicit `dyn` are deprecated
        //[e2018]~^^^^ WARN: this is accepted in the current edition
        //[e2021]~^^^^^ ERROR: expected a type, found a trait
    }
}

pub mod z {}

fn main() {}
