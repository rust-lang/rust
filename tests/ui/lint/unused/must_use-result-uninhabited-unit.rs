//@ edition: 2024
//@ aux-crate:dep=must_use_result_unit_uninhabited_extern_crate.rs

#![deny(unused_must_use)]
#![feature(never_type)]

use core::ops::{ControlFlow, ControlFlow::Break};
use dep::{MyUninhabited, MyUninhabitedNonexhaustive};

fn result_unit_unit() -> Result<(), ()> {
    Err(())
}

fn result_infallible_unit() -> Result<core::convert::Infallible, ()> {
    Err(())
}

fn result_never_unit() -> Result<!, ()> {
    Err(())
}

fn result_myuninhabited_unit() -> Result<MyUninhabited, ()> {
    Err(())
}

fn result_myuninhabited_nonexhaustive_unit() -> Result<MyUninhabitedNonexhaustive, ()> {
    Err(())
}

trait AssocType {
    type Error;
}

struct S1;
impl AssocType for S1 {
    type Error = !;
}

struct S2;
impl AssocType for S2 {
    type Error = ();
}

fn result_unit_assoctype<AT: AssocType>(_: AT) -> Result<AT::Error, ()> {
    Err(())
}

trait UsesAssocType {
    type Error;
    fn method_use_assoc_type(&self) -> Result<Self::Error, ()>;
}

impl UsesAssocType for S1 {
    type Error = !;
    fn method_use_assoc_type(&self) -> Result<Self::Error, ()> {
        Err(())
    }
}

impl UsesAssocType for S2 {
    type Error = ();
    fn method_use_assoc_type(&self) -> Result<Self::Error, ()> {
        Ok(())
    }
}

fn controlflow_unit_unit() -> ControlFlow<()> {
    Break(())
}

fn controlflow_unit_infallible() -> ControlFlow<(), core::convert::Infallible> {
    Break(())
}

fn controlflow_unit_never() -> ControlFlow<(), !> {
    Break(())
}

fn main() {
    result_unit_unit(); //~ ERROR: unused `Result` that must be used
    result_infallible_unit();
    result_never_unit();
    result_myuninhabited_unit();
    result_myuninhabited_nonexhaustive_unit(); //~ ERROR: unused `Result` that must be used
    result_unit_assoctype(S1);
    result_unit_assoctype(S2); //~ ERROR: unused `Result` that must be used
    S1.method_use_assoc_type();
    S2.method_use_assoc_type(); //~ ERROR: unused `Result` that must be used

    controlflow_unit_unit(); //~ ERROR: unused `ControlFlow` that must be used
    controlflow_unit_infallible();
    controlflow_unit_never();
}

trait AssocTypeBeforeMonomorphisation {
    type Error;
    fn generate(&self) -> Result<Self::Error, ()>;
    fn process(&self) {
        self.generate(); //~ ERROR: unused `Result` that must be used
    }
}
