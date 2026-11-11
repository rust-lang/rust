#![crate_type = "lib"]

#[diagnostic::on_unknown(message = "oh no, that is not in aux module `{this}`")]
#[path = "auxiliary/module.rs"]
pub mod x;

pub use x::Bar;
//~^ ERROR oh no, that is not in aux module `x`
