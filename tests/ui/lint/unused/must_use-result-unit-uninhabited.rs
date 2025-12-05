//@ edition: 2024
//@ aux-crate:dep=must_use_result_unit_uninhabited_extern_crate.rs

#![deny(unused_must_use)]
#![feature(never_type)]

use core::ops::{ControlFlow, ControlFlow::Continue};
use dep::{MyUninhabited, MyUninhabitedNonexhaustive};

fn result_unit_unit() -> Result<(), ()> {
    Ok(())
}

fn result_unit_infallible() -> Result<(), core::convert::Infallible> {
    Ok(())
}

fn result_unit_never() -> Result<(), !> {
    Ok(())
}

fn result_unit_myuninhabited() -> Result<(), MyUninhabited> {
    Ok(())
}

fn result_unit_myuninhabited_nonexhaustive() -> Result<(), MyUninhabitedNonexhaustive> {
    Ok(())
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

fn result_unit_assoctype<AT: AssocType>(_: AT) -> Result<(), AT::Error> {
    Ok(())
}

trait UsesAssocType {
    type Error;
    fn method_use_assoc_type(&self) -> Result<(), Self::Error>;
}

impl UsesAssocType for S1 {
    type Error = !;
    fn method_use_assoc_type(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl UsesAssocType for S2 {
    type Error = ();
    fn method_use_assoc_type(&self) -> Result<(), Self::Error> {
        Err(())
    }
}

fn controlflow_unit() -> ControlFlow<()> {
    Continue(())
}

fn controlflow_infallible_unit() -> ControlFlow<core::convert::Infallible, ()> {
    Continue(())
}

fn controlflow_never() -> ControlFlow<!> {
    Continue(())
}

fn main() {
    result_unit_unit(); //~ ERROR: unused `Result` that must be used
    result_unit_infallible();
    result_unit_never();
    result_unit_myuninhabited();
    result_unit_myuninhabited_nonexhaustive(); //~ ERROR: unused `Result` that must be used
    result_unit_assoctype(S1);
    result_unit_assoctype(S2); //~ ERROR: unused `Result` that must be used
    S1.method_use_assoc_type();
    S2.method_use_assoc_type(); //~ ERROR: unused `Result` that must be used

    controlflow_unit(); //~ ERROR: unused `ControlFlow` that must be used
    controlflow_infallible_unit();
    controlflow_never();
}

trait AssocTypeBeforeMonomorphisation {
    type Error;
    fn generate(&self) -> Result<(), Self::Error>;
    fn process(&self) {
        self.generate(); //~ ERROR: unused `Result` that must be used
    }
}
