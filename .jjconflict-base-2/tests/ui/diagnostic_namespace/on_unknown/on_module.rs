#![crate_type = "lib"]
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(message = "oh no, `{Unresolved}` is not in module `{This}`")]
pub mod x {
    pub struct Foo;
}

pub use x::Bar;
//~^ ERROR oh no, `Bar` is not in module `x`
