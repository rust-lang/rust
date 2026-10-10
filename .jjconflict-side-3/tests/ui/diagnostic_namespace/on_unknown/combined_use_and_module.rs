#![crate_type = "lib"]
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(
    message = "THIS MUST NEVER BE SHOWN",
    label = "THIS MUST NEVER BE SHOWN",
    note = "module note 1",
    note = "module note 2"
)]
pub mod constants {
    pub const ONE: usize = 1;
    pub const TWO: usize = 3;
    pub const FOUR: usize = 4;
}

#[diagnostic::on_unknown(
    message = "the message",
    label = "the label",
    note = "use note 1",
    note = "use note 2"
)]
pub use constants::THREE;
//~^ ERROR the message
//~| NOTE the label
//~| NOTE unresolved import `constants::THREE`
//~| NOTE use note 1
//~| NOTE use note 2
//~| NOTE module note 1
//~| NOTE module note 2
