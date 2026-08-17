//@ edition: 2024

#![feature(edition_redirect)]
#![allow(internal_features)]

pub struct Old;
pub struct Current;

#[rustc_edition_redirect = "2021"]
pub use Old as Name;
pub use Current as Name;

mod diagnostic_targets {
    #[doc(alias = "OldAlias")]
    pub struct AliasCarrier;

    pub trait Candidate {}

    pub mod diagnostic_module {
        pub enum DiagnosticEnum {
            Variant(u8),
        }
    }
}

#[rustc_edition_redirect = "2021"]
pub use diagnostic_targets::AliasCarrier as AliasCarrier;
pub struct AliasCarrier;

#[rustc_edition_redirect = "2021"]
pub use diagnostic_targets::Candidate as Candidate;
pub struct Candidate;

#[rustc_edition_redirect = "2021"]
pub use diagnostic_targets::diagnostic_module as diagnostic_module;
pub mod diagnostic_module {
    pub enum DiagnosticEnum {
        CurrentVariant(u8),
    }
}

pub mod trait_prelude {
    pub struct OldItem;
    pub struct CurrentItem;

    #[rustc_edition_redirect = "2021"]
    pub use OldItem as RedirectedItem;
    pub use CurrentItem as RedirectedItem;

    pub struct OldMarker;
    pub struct CurrentMarker;

    mod old {
        pub trait RedirectedTrait {
            fn redirected_method(&self) -> super::OldMarker;
        }

        impl RedirectedTrait for () {
            fn redirected_method(&self) -> super::OldMarker {
                super::OldMarker
            }
        }
    }

    #[rustc_edition_redirect = "2021"]
    pub use old::RedirectedTrait as RedirectedTrait;

    pub trait RedirectedTrait {
        fn redirected_method(&self) -> CurrentMarker;
    }

    impl RedirectedTrait for () {
        fn redirected_method(&self) -> CurrentMarker {
            CurrentMarker
        }
    }
}

#[macro_export]
macro_rules! old_macro {
    () => {
        pub type Selected = $crate::Old;
    };
}

#[rustc_edition_redirect = "2021"]
pub use old_macro as redirected_macro;

#[macro_export]
macro_rules! redirected_macro {
    () => {
        pub type Selected = $crate::Current;
    };
}

pub mod nested {}
