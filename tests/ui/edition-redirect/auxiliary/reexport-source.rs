//@ edition: 2024

#![feature(edition_redirect)]

pub struct Old;

#[rustc_edition_redirect = "2021"]
pub use Old as Current;
pub struct Current;

pub fn old() -> Old {
    Old
}

pub fn current() -> Current {
    Current
}

pub mod old_module {
    pub struct Child;
}

#[rustc_edition_redirect = "2021"]
pub use old_module as redirected_module;
pub mod redirected_module {
    pub struct Child;
}

pub fn old_child() -> old_module::Child {
    old_module::Child
}

pub fn current_child() -> redirected_module::Child {
    redirected_module::Child
}
