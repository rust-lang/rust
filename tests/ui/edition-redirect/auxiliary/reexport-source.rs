//@ edition: 2024

#![feature(edition_redirect)]

pub struct Old;

#[rustc_edition_redirect(before = "2024", target(Old))]
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

#[rustc_edition_redirect(before = "2024", target(old_module))]
pub mod redirected_module {
    pub struct Child;
}

pub fn old_child() -> old_module::Child {
    old_module::Child
}

pub fn current_child() -> redirected_module::Child {
    redirected_module::Child
}
