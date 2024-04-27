//@ check-pass

#![feature(rustc_attrs)]
#![allow(unused)]

use std::borrow::Borrow;
use std::ops::Deref;

struct PlainType<T>(T);

#[derive(Clone)]
struct CloneType<T>(T);

fn check(mut encoded: &[u8]) {
    let _ = &mut encoded.clone();
    //~^ WARN call to `.clone()` on a reference in this situation does nothing
    let _ = &encoded.clone();
    //~^ WARN call to `.clone()` on a reference in this situation does nothing
}

fn main() {
    let non_clone_type_ref = &PlainType(1u32);
    let non_clone_type_ref_clone: &PlainType<u32> = non_clone_type_ref.clone();
    //~^ WARN call to `.clone()` on a reference in this situation does nothing

    let clone_type_ref = &CloneType(1u32);
    let clone_type_ref_clone: CloneType<u32> = clone_type_ref.clone();


    let non_deref_type = &PlainType(1u32);
    let non_deref_type_deref: &PlainType<u32> = non_deref_type.deref();
    //~^ WARN call to `.deref()` on a reference in this situation does nothing

    let non_borrow_type = &PlainType(1u32);
    let non_borrow_type_borrow: &PlainType<u32> = non_borrow_type.borrow();
    //~^ WARN call to `.borrow()` on a reference in this situation does nothing

    // Borrowing a &&T does not warn since it has collapsed the double reference
    let non_borrow_type = &&PlainType(1u32);
    let non_borrow_type_borrow: &PlainType<u32> = non_borrow_type.borrow();
}

fn generic<T>(non_clone_type: &PlainType<T>) {
    non_clone_type.clone();
    //~^ WARN call to `.clone()` on a reference in this situation does nothing
}

fn non_generic(non_clone_type: &PlainType<u32>) {
    non_clone_type.clone();
    //~^ WARN call to `.clone()` on a reference in this situation does nothing
}

struct DiagnosticClone;
impl Clone for DiagnosticClone {
    #[rustc_diagnostic_item = "other_clone"]
    fn clone(&self) -> Self {
        DiagnosticClone
    }
}

fn with_other_diagnostic_item(x: DiagnosticClone) {
    x.clone();
}
