#![crate_type = "lib"]

#[diagnostic::on_unknown(message = "oh no, that is not in aux module `{This}`")]
#[path = "auxiliary/module.rs"]
pub mod x;

pub use x::Bar;
//~^ ERROR oh no, that is not in aux module `x`
