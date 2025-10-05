//@ edition: 2024
//@ aux-crate:dep=must_use_result_unit_uninhabited_extern_crate.rs

#![deny(unused_must_use)]
#![feature(never_type)]

use dep::{MyUninhabited, MyUninhabitedNonexhaustive};

fn f1() -> Result<(), ()> {
    Ok(())
}

fn f2() -> Result<(), core::convert::Infallible> {
    Ok(())
}

fn f3() -> Result<(), !> {
    Ok(())
}

fn f4() -> Result<(), MyUninhabited> {
    Ok(())
}

fn f5() -> Result<(), MyUninhabitedNonexhaustive> {
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

fn f6<AT: AssocType>(_: AT) -> Result<(), AT::Error> {
    Ok(())
}

fn main() {
    f1(); //~ ERROR: unused `Result` that must be used
    f2();
    f3();
    f4();
    f5(); //~ ERROR: unused `Result` that must be used
    f6(S1);
    f6(S2); //~ ERROR: unused `Result` that must be used
}
