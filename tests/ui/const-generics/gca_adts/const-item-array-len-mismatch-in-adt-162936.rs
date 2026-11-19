//! Regression test for <https://github.com/rust-lang/rust/issues/162936>.
#![feature(gca_macroless_items, gca_adts, gca_min_const_items, min_adt_const_params)]

pub struct Data([u8; 3]);

const DATA: Data = Data([1, 2, 3, 4]);
//~^ ERROR `Data` must implement `ConstParamTy` to be used as the type of a const generic parameter
//~| ERROR the constant `*b"\x01\x02\x03\x04"` is not of type `[u8; 3]`

fn main() {
    match DATA {
        _ => (),
    }
}
