#![crate_type = "lib"]

#[diagnostic::on_unknown(message = "oh no, `{unresolved}` is not in module `{this}`")]
pub mod x {
    pub struct Foo;
}

pub use x::Bar;
//~^ ERROR oh no, `Bar` is not in module `x`
