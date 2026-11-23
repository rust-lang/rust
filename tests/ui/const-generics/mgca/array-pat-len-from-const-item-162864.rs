//! Regression test for <https://github.com/rust-lang/rust/issues/162864>.
#![feature(gca_min_const_items, gca_macroless_items, min_adt_const_params)]

use std::marker::ConstParamTy;

const N: usize = todo!();
//~^ ERROR evaluation panicked: not yet implemented

#[derive(Eq, PartialEq, ConstParamTy)]
struct Data([u8; N]);

const DATA: Data = Data([1, 2, 3, 4]);

fn main() {
    match DATA {
        DATA | _ => {}
    }
}
