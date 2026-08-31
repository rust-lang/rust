#![feature(edition_redirect)]

pub struct Oldest;
pub struct Middle;

#[rustc_edition_redirect = "2021"]
pub use Middle as Redirected;
#[rustc_edition_redirect = "2018"]
pub use Oldest as Redirected;
pub struct Redirected;

pub mod oldest_module {
    pub const VALUE: usize = 1;
}

pub mod middle_module {
    pub const VALUE: usize = 2;
}

#[rustc_edition_redirect = "2018"]
pub use oldest_module as redirected_module;
#[rustc_edition_redirect = "2021"]
pub use middle_module as redirected_module;
pub mod redirected_module {
    pub const VALUE: usize = 3;
}

pub mod use_targets {
    pub struct OldestUse;
    pub struct MiddleUse;
    pub struct CurrentUse;
}

#[rustc_edition_redirect = "2018"]
pub use use_targets::OldestUse as RedirectedUse;
#[rustc_edition_redirect = "2021"]
pub use use_targets::MiddleUse as RedirectedUse;
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

    #[rustc_edition_redirect = "2021"]
    pub use self::OldAlias as Redirected;
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

#[rustc_edition_redirect = "2018"]
pub use oldest_macro as redirected_macro;
#[rustc_edition_redirect = "2021"]
pub use middle_macro as redirected_macro;
#[macro_export]
macro_rules! redirected_macro {
    () => { 3 };
}

pub mod ambiguity {
    pub struct Shared;
    pub struct OldA;
    pub struct OldB;

    pub mod alias_a {
        #[rustc_edition_redirect = "2021"]
        pub use super::OldA as Item;
        pub use super::Shared as Item;
    }

    pub mod alias_b {
        #[rustc_edition_redirect = "2021"]
        pub use super::OldB as Item;
        pub use super::Shared as Item;
    }
}

fn local_resolution_uses_default_items() {
    let _: Redirected = Redirected;
    let _: use_targets::CurrentUse = RedirectedUse;
    let _: reexport_scope::Current = reexport_scope::Redirected;
    const _: [(); 3] = [(); redirected_module::VALUE];
    const _: [(); 3] = [(); redirected_macro!()];
}
