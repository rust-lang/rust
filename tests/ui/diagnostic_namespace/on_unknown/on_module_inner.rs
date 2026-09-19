#![crate_type = "lib"]
#![feature(diagnostic_on_unknown)]
#![feature(custom_inner_attributes)]

pub mod x {
    #![diagnostic::on_unknown(message = "oh no, `{Unresolved}` is not in module `{This}`")]
    pub struct Foo;
}

pub use x::Bar;
//~^ ERROR oh no, `Bar` is not in module `x`
