#![crate_type = "lib"]
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(
    message = "NEVER SHOWN",
    label = "label different",
    note = "note different"
)]
pub mod different {}

#[diagnostic::on_unknown(message = "the same message", label = "label x", note = "note x")]
pub mod x {}

#[diagnostic::on_unknown(label = "label y", note = "note y")]
pub mod y {}

#[diagnostic::on_unknown(message = "the same message", note = "note z")]
pub mod z {}

pub use {
    x::Bar,
    //~^ ERROR the same message
    //~| NOTE note x
    //~| NOTE note y
    //~| NOTE note z
    //~| NOTE unresolved imports `x::Bar`, `y::Foo`, `z::Baz`
    //~| NOTE label x
    y::Foo,
    //~^ NOTE label y
    z::Baz,
    //~^ NOTE no `Baz` in `z`
};

pub use {
    different::nothing,
    //~^ ERROR unresolved imports `different::nothing`, `x::Buz`
    //~| NOTE note x
    //~| NOTE note different
    //~| NOTE label different
    x::Buz,
    //~^ NOTE label x
};
