#![feature(edition_redirect)]

pub struct Oldest;
pub struct Middle;

#[rustc_edition_redirect(before = "2024", target(Middle))]
#[rustc_edition_redirect(before = "2021", target(Oldest))]
pub struct Redirected;

pub mod oldest_module {
    pub const VALUE: usize = 1;
}

pub mod middle_module {
    pub const VALUE: usize = 2;
}

#[rustc_edition_redirect(before = "2021", target(oldest_module))]
#[rustc_edition_redirect(before = "2024", target(middle_module))]
pub mod redirected_module {
    pub const VALUE: usize = 3;
}

pub mod use_targets {
    pub struct OldestUse;
    pub struct MiddleUse;
    pub struct CurrentUse;
}

#[rustc_edition_redirect(before = "2021", target(use_targets::OldestUse))]
#[rustc_edition_redirect(before = "2024", target(use_targets::MiddleUse))]
pub use use_targets::CurrentUse as RedirectedUse;

pub mod same_redirect_a {
    pub use crate::RedirectedUse as Item;
}

pub mod same_redirect_b {
    pub use crate::RedirectedUse as Item;
}

pub mod same_redirects {
    pub use crate::same_redirect_a::*;
    pub use crate::same_redirect_b::*;
}

pub mod reexport_scope {
    pub struct Old;
    pub struct Current;

    #[rustc_edition_redirect(before = "2024", target(OldAlias))]
    pub use self::Current as Redirected;

    pub use self::Old as OldAlias;
}

pub use reexport_scope::Redirected as ScopedRedirected;

#[macro_export]
macro_rules! oldest_macro {
    () => { 1 };
}

#[macro_export]
macro_rules! middle_macro {
    () => { 2 };
}

#[rustc_edition_redirect(before = "2021", target(oldest_macro))]
#[rustc_edition_redirect(before = "2024", target(middle_macro))]
#[macro_export]
macro_rules! redirected_macro {
    () => { 3 };
}

pub mod ambiguity {
    pub struct Shared;
    pub struct OldA;
    pub struct OldB;

    pub mod alias_a {
        #[rustc_edition_redirect(before = "2024", target(super::OldA))]
        pub use super::Shared as Item;
    }

    pub mod alias_b {
        #[rustc_edition_redirect(before = "2024", target(super::OldB))]
        pub use super::Shared as Item;
    }
}
