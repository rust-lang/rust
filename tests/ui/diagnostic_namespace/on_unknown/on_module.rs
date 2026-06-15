#![crate_type = "lib"]
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(message = "oh no, that is not in module `{This}`")]
pub mod x {
    pub struct Foo;
}

pub use x::Bar;
//~^ ERROR oh no, that is not in module `x`
