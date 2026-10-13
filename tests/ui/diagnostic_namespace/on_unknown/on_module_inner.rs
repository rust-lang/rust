#![crate_type = "lib"]
#![feature(custom_inner_attributes)]

pub mod x {
    #![diagnostic::on_unknown(message = "oh no, `{Unresolved}` is not in module `{This}`")]
    pub struct Foo;
}

pub use x::Bar;
//~^ ERROR oh no, `Bar` is not in module `x`
