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
}

pub mod default {
    pub trait Default {
        fn default() -> Self;
    }
}

pub mod iter {
    pub use self::traits::{collect::IntoIterator, iterator::Iterator};
    mod traits {
        pub(crate) mod iterator {
            use crate::option::Option;
            pub trait Iterator {
                type Item;
                fn next(&mut self) -> Option<Self::Item>;
                fn by_ref(&mut self) -> &mut Self {
                    self
                }
                fn take(self, n: usize) -> crate::iter::Take<Self> {
                    crate::iter::Take { inner: self }
                }
            }

            impl<I: Iterator> Iterator for &mut I {
                type Item = I::Item;
                fn next(&mut self) -> Option<I::Item> {
                    (**self).next()
                }
            }
        }
        pub(crate) mod collect {
            pub trait IntoIterator {
                type Item;
            }
        }
    }

    pub use self::sources::*;
    pub(crate) mod sources {
        use super::Iterator;
        use crate::option::Option::{self, *};
        pub struct Repeat<A> {
            element: A,
        }

        pub fn repeat<T>(elt: T) -> Repeat<T> {
            Repeat { element: elt }
        }

        impl<A> Iterator for Repeat<A> {
            type Item = A;

            fn next(&mut self) -> Option<A> {
                None
            }
        }
    }

    pub use self::adapters::*;
    pub(crate) mod adapters {
        use super::Iterator;
        use crate::option::Option::{self, *};
        pub struct Take<I> {
            pub(crate) inner: I,
        }
        impl<I> Iterator for Take<I>
        where
            I: Iterator,
        {
            type Item = <I as Iterator>::Item;
            fn next(&mut self) -> Option<<I as Iterator>::Item> {
                None
            }
        }
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
}

pub mod option {
    pub enum Option<T> {
        None,
        Some(T),
    }
}

pub mod prelude {
    pub use crate::{
        cmp::Ord,
        convert::From,
        default::Default,
        iter::{IntoIterator, Iterator},
        ops::{Fn, FnMut, FnOnce},
        option::Option::{self, *},
    };
}
#[prelude_import]
pub use prelude::*;
//- /libstd.rs crate:std deps:core
//! Signatures of traits, types and functions from the std lib for use in tests.

/// Docs for return_keyword
mod return_keyword {}

/// Docs for prim_str
mod prim_str {}
