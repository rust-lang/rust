// Ensure that we properly print the value `1` as `1` in the initializer of associated constants
// that have user type "projection".
//
// We once used to evaluate the initializer in rustdoc and use rustc's MIR pretty-printer to
// render the resulting MIR const value. This pretty printer matches on the type to interpret
// the data and falls back to a cryptic `"{transmute(0x$data): $ty}"` for types it can't handle.
// Crucially, when constructing the MIR const we passed the unnormalized type of the initializer,
// i.e., the projection `<Struct as Trait>::Ty` instead of the normalized `u32` which the
// pretty printer obviously can't handle.
//
// Now we no longer evaluate it and use a custom printer for the const expr.
//
// issue: <https://github.com/rust-lang/rust/issues/150312>

#![crate_name = "it"]

pub trait Trait {
    type Ty;

    const CT: Self::Ty;
}

pub struct Struct;

impl Trait for Struct {
    type Ty = u32;

    //@ has it/struct.Struct.html
    //@ has - '//*[@id="associatedconstant.CT"]' 'const CT: Self::Ty = 1'
    const CT: Self::Ty = 1;
}
