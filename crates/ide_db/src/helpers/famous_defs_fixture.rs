//- /libcore.rs crate:core
//! Signatures of traits, types and functions from the core lib for use in tests.
pub mod cmp {

    pub trait Ord {
        fn cmp(&self, other: &Self) -> Ordering;
        fn max(self, other: Self) -> Self;
        fn min(self, other: Self) -> Self;
        fn clamp(self, min: Self, max: Self) -> Self;
    }
}

pub mod convert {
    pub trait From<T> {
        fn from(t: T) -> Self;
    }

    pub trait Into<T> {
        pub fn into(self) -> T;
    }
}

pub mod default {
    pub trait Default {
        fn default() -> Self;
    }
}

pub mod ops {
    #[lang = "fn"]
    pub trait Fn<Args>: FnMut<Args> {
        extern "rust-call" fn call(&self, args: Args) -> Self::Output;
    }

    #[lang = "fn_mut"]
    pub trait FnMut<Args>: FnOnce<Args> {
        extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
    }
    #[lang = "fn_once"]
    pub trait FnOnce<Args> {
        #[lang = "fn_once_output"]
        type Output;
        extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
    }

    #[lang = "deref"]
    pub trait Deref {
        type Target: ?Sized;
        fn deref(&self) -> &Self::Target;
    }
}

pub mod option {
    pub enum Option<T> {
        None,
        Some(T),
    }
}

pub mod prelude {
    pub mod rust_2018 {
        pub use crate::{
            cmp::Ord,
            convert::{From, Into},
            default::Default,
            iter::{IntoIterator, Iterator},
            ops::{Fn, FnMut, FnOnce},
            option::Option::{self, *},
        };
    }
}
#[prelude_import]
pub use prelude::rust_2018::*;
//- /libstd.rs crate:std deps:core
//! Signatures of traits, types and functions from the std lib for use in tests.

/// Docs for return_keyword
mod return_keyword {}

/// Docs for prim_str
mod prim_str {}

pub use core::ops;
