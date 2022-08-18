//! This is a fixture we use for tests that need lang items.
//!
//! We want to include the minimal subset of core for each test, so this file
//! supports "conditional compilation". Tests use the following syntax to include minicore:
//!
//!  //- minicore: flag1, flag2
//!
//! We then strip all the code marked with other flags.
//!
//! Available flags:
//!     sized:
//!     unsize: sized
//!     coerce_unsized: unsize
//!     slice:
//!     range:
//!     deref: sized
//!     deref_mut: deref
//!     index: sized
//!     fn:
//!     try:
//!     pin:
//!     future: pin
//!     option:
//!     result:
//!     iterator: option
//!     iterators: iterator, fn
//!     default: sized
//!     hash:
//!     clone: sized
//!     copy: clone
//!     from: sized
//!     eq: sized
//!     ord: eq, option
//!     derive:
//!     fmt: result
//!     bool_impl: option, fn
//!     add:
//!     as_ref: sized
//!     drop:

pub mod marker {
    // region:sized
    #[lang = "sized"]
    #[fundamental]
    #[rustc_specialization_trait]
    pub trait Sized {}
    // endregion:sized

    // region:unsize
    #[lang = "unsize"]
    pub trait Unsize<T: ?Sized> {}
    // endregion:unsize

    // region:copy
    #[lang = "copy"]
    pub trait Copy: Clone {}
    // region:derive
    #[rustc_builtin_macro]
    pub macro Copy($item:item) {}
    // endregion:derive

    mod copy_impls {
        use super::Copy;

        macro_rules! impl_copy {
            ($($t:ty)*) => {
                $(
                    impl Copy for $t {}
                )*
            }
        }

        impl_copy! {
            usize u8 u16 u32 u64 u128
            isize i8 i16 i32 i64 i128
            f32 f64
            bool char
        }

        impl<T: ?Sized> Copy for *const T {}
        impl<T: ?Sized> Copy for *mut T {}
        impl<T: ?Sized> Copy for &T {}
    }
    // endregion:copy
}

// region:default
pub mod default {
    pub trait Default: Sized {
        fn default() -> Self;
    }
    // region:derive
    #[rustc_builtin_macro]
    pub macro Default($item:item) {}
    // endregion:derive
}
// endregion:default

// region:hash
pub mod hash {
    pub trait Hasher {}

    pub trait Hash {
        fn hash<H: Hasher>(&self, state: &mut H);
    }
}
// endregion:hash

// region:clone
pub mod clone {
    #[lang = "clone"]
    pub trait Clone: Sized {
        fn clone(&self) -> Self;
    }
    // region:derive
    #[rustc_builtin_macro]
    pub macro Clone($item:item) {}
    // endregion:derive
}
// endregion:clone

pub mod convert {
    // region:from
    pub trait From<T>: Sized {
        fn from(_: T) -> Self;
    }
    pub trait Into<T>: Sized {
        fn into(self) -> T;
    }

    impl<T, U> Into<U> for T
    where
        U: From<T>,
    {
        fn into(self) -> U {
            U::from(self)
        }
    }

    impl<T> From<T> for T {
        fn from(t: T) -> T {
            t
        }
    }
    // endregion:from

    // region:as_ref
    pub trait AsRef<T: ?Sized> {
        fn as_ref(&self) -> &T;
    }
    // endregion:as_ref
}

pub mod ops {
    // region:coerce_unsized
    mod unsize {
        use crate::marker::Unsize;

        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T: ?Sized> {}

        impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a mut U> for &'a mut T {}
        impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b mut T {}
        impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for &'a mut T {}
        impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a mut T {}

        impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
        impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for &'a T {}

        impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}
        impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *mut T {}
        impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *const T {}
    }
    pub use self::unsize::CoerceUnsized;
    // endregion:coerce_unsized

    // region:deref
    mod deref {
        #[lang = "deref"]
        pub trait Deref {
            #[lang = "deref_target"]
            type Target: ?Sized;
            fn deref(&self) -> &Self::Target;
        }
        // region:deref_mut
        #[lang = "deref_mut"]
        pub trait DerefMut: Deref {
            fn deref_mut(&mut self) -> &mut Self::Target;
        }
        // endregion:deref_mut
    }
    pub use self::deref::{
        Deref,
        DerefMut, // :deref_mut
    };
    // endregion:deref

    // region:drop
    #[lang = "drop"]
    pub trait Drop {
        fn drop(&mut self);
    }
    // endregion:drop

    // region:index
    mod index {
        #[lang = "index"]
        pub trait Index<Idx: ?Sized> {
            type Output: ?Sized;
            fn index(&self, index: Idx) -> &Self::Output;
        }
        #[lang = "index_mut"]
        pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
            fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
        }

        // region:slice
        impl<T, I> Index<I> for [T]
        where
            I: SliceIndex<[T]>,
        {
            type Output = I::Output;
            fn index(&self, index: I) -> &I::Output {
                loop {}
            }
        }
        impl<T, I> IndexMut<I> for [T]
        where
            I: SliceIndex<[T]>,
        {
            fn index_mut(&mut self, index: I) -> &mut I::Output {
                loop {}
            }
        }

        pub unsafe trait SliceIndex<T: ?Sized> {
            type Output: ?Sized;
        }
        unsafe impl<T> SliceIndex<[T]> for usize {
            type Output = T;
        }
        // endregion:slice
    }
    pub use self::index::{Index, IndexMut};
    // endregion:index

    // region:drop
    pub mod mem {
        pub fn drop<T>(_x: T) {}
    }
    // endregion:drop

    // region:range
    mod range {
        #[lang = "RangeFull"]
        pub struct RangeFull;

        #[lang = "Range"]
        pub struct Range<Idx> {
            pub start: Idx,
            pub end: Idx,
        }

        #[lang = "RangeFrom"]
        pub struct RangeFrom<Idx> {
            pub start: Idx,
        }

        #[lang = "RangeTo"]
        pub struct RangeTo<Idx> {
            pub end: Idx,
        }

        #[lang = "RangeInclusive"]
        pub struct RangeInclusive<Idx> {
            pub(crate) start: Idx,
            pub(crate) end: Idx,
            pub(crate) exhausted: bool,
        }

        #[lang = "RangeToInclusive"]
        pub struct RangeToInclusive<Idx> {
            pub end: Idx,
        }
    }
    pub use self::range::{Range, RangeFrom, RangeFull, RangeTo};
    pub use self::range::{RangeInclusive, RangeToInclusive};
    // endregion:range

    // region:fn
    mod function {
        #[lang = "fn"]
        #[fundamental]
        pub trait Fn<Args>: FnMut<Args> {}

        #[lang = "fn_mut"]
        #[fundamental]
        pub trait FnMut<Args>: FnOnce<Args> {}

        #[lang = "fn_once"]
        #[fundamental]
        pub trait FnOnce<Args> {
            #[lang = "fn_once_output"]
            type Output;
        }
    }
    pub use self::function::{Fn, FnMut, FnOnce};
    // endregion:fn
    // region:try
    mod try_ {
        pub enum ControlFlow<B, C = ()> {
            Continue(C),
            Break(B),
        }
        pub trait FromResidual<R = Self::Residual> {
            #[lang = "from_residual"]
            fn from_residual(residual: R) -> Self;
        }
        #[lang = "try"]
        pub trait Try: FromResidual<Self::Residual> {
            type Output;
            type Residual;
            #[lang = "from_output"]
            fn from_output(output: Self::Output) -> Self;
            #[lang = "branch"]
            fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
        }

        impl<B, C> Try for ControlFlow<B, C> {
            type Output = C;
            type Residual = ControlFlow<B, convert::Infallible>;
            fn from_output(output: Self::Output) -> Self {}
            fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {}
        }

        impl<B, C> FromResidual for ControlFlow<B, C> {
            fn from_residual(residual: ControlFlow<B, convert::Infallible>) -> Self {}
        }
    }
    pub use self::try_::{ControlFlow, FromResidual, Try};
    // endregion:try

    // region:add
    #[lang = "add"]
    pub trait Add<Rhs = Self> {
        type Output;
        fn add(self, rhs: Rhs) -> Self::Output;
    }
    // endregion:add
}

// region:eq
pub mod cmp {
    #[lang = "eq"]
    pub trait PartialEq<Rhs: ?Sized = Self> {
        fn eq(&self, other: &Rhs) -> bool;
        fn ne(&self, other: &Rhs) -> bool {
            !self.eq(other)
        }
    }

    pub trait Eq: PartialEq<Self> {}

    // region:derive
    #[rustc_builtin_macro]
    pub macro PartialEq($item:item) {}
    #[rustc_builtin_macro]
    pub macro Eq($item:item) {}
    // endregion:derive

    // region:ord
    #[lang = "partial_ord"]
    pub trait PartialOrd<Rhs: ?Sized = Self>: PartialEq<Rhs> {
        fn partial_cmp(&self, other: &Rhs) -> Option<Ordering>;
    }

    pub trait Ord: Eq + PartialOrd<Self> {
        fn cmp(&self, other: &Self) -> Ordering;
    }

    pub enum Ordering {
        Less = -1,
        Equal = 0,
        Greater = 1,
    }

    // region:derive
    #[rustc_builtin_macro]
    pub macro PartialOrd($item:item) {}
    #[rustc_builtin_macro]
    pub macro Ord($item:item) {}
    // endregion:derive

    // endregion:ord
}
// endregion:eq

// region:fmt
pub mod fmt {
    pub struct Error;
    pub type Result = Result<(), Error>;
    pub struct Formatter<'a>;
    pub trait Debug {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result;
    }
}
// endregion:fmt

// region:slice
pub mod slice {
    #[lang = "slice"]
    impl<T> [T] {
        pub fn len(&self) -> usize {
            loop {}
        }
    }
}
// endregion:slice

// region:option
pub mod option {
    pub enum Option<T> {
        #[lang = "None"]
        None,
        #[lang = "Some"]
        Some(T),
    }

    impl<T> Option<T> {
        pub const fn unwrap(self) -> T {
            match self {
                Some(val) => val,
                None => panic!("called `Option::unwrap()` on a `None` value"),
            }
        }
    }
}
// endregion:option

// region:result
pub mod result {
    pub enum Result<T, E> {
        #[lang = "Ok"]
        Ok(T),
        #[lang = "Err"]
        Err(E),
    }
}
// endregion:result

// region:pin
pub mod pin {
    #[lang = "pin"]
    #[fundamental]
    pub struct Pin<P> {
        pointer: P,
    }
}
// endregion:pin

// region:future
pub mod future {
    use crate::{
        pin::Pin,
        task::{Context, Poll},
    };

    #[lang = "future_trait"]
    pub trait Future {
        type Output;
        #[lang = "poll"]
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
    }

    pub trait IntoFuture {
        type Output;
        type IntoFuture: Future<Output = Self::Output>;
        #[lang = "into_future"]
        fn into_future(self) -> Self::IntoFuture;
    }

    impl<F: Future> IntoFuture for F {
        type Output = F::Output;
        type IntoFuture = F;
        fn into_future(self) -> F {
            self
        }
    }
}
pub mod task {
    pub enum Poll<T> {
        #[lang = "Ready"]
        Ready(T),
        #[lang = "Pending"]
        Pending,
    }

    pub struct Context<'a> {
        waker: &'a (),
    }
}
// endregion:future

// region:iterator
pub mod iter {
    // region:iterators
    mod adapters {
        pub struct Take<I> {
            iter: I,
            n: usize,
        }
        impl<I> Iterator for Take<I>
        where
            I: Iterator,
        {
            type Item = <I as Iterator>::Item;

            fn next(&mut self) -> Option<<I as Iterator>::Item> {
                loop {}
            }
        }

        pub struct FilterMap<I, F> {
            iter: I,
            f: F,
        }
        impl<B, I: Iterator, F> Iterator for FilterMap<I, F>
        where
            F: FnMut(I::Item) -> Option<B>,
        {
            type Item = B;

            #[inline]
            fn next(&mut self) -> Option<B> {
                loop {}
            }
        }
    }
    pub use self::adapters::{Take, FilterMap};

    mod sources {
        mod repeat {
            pub fn repeat<T>(elt: T) -> Repeat<T> {
                loop {}
            }

            pub struct Repeat<A> {
                element: A,
            }

            impl<A> Iterator for Repeat<A> {
                type Item = A;

                fn next(&mut self) -> Option<A> {
                    loop {}
                }
            }
        }
        pub use self::repeat::{repeat, Repeat};
    }
    pub use self::sources::{repeat, Repeat};
    // endregion:iterators

    mod traits {
        mod iterator {
            use super::super::Take;

            pub trait Iterator {
                type Item;
                #[lang = "next"]
                fn next(&mut self) -> Option<Self::Item>;
                fn nth(&mut self, n: usize) -> Option<Self::Item> {
                    loop {}
                }
                fn by_ref(&mut self) -> &mut Self
                where
                    Self: Sized,
                {
                    self
                }
                // region:iterators
                fn take(self, n: usize) -> crate::iter::Take<Self> {
                    loop {}
                }
                fn filter_map<B, F>(self, f: F) -> crate::iter::FilterMap<Self, F>
                where
                    Self: Sized,
                    F: FnMut(Self::Item) -> Option<B>,
                {
                    loop {}
                }
                // endregion:iterators
            }
            impl<I: Iterator + ?Sized> Iterator for &mut I {
                type Item = I::Item;
                fn next(&mut self) -> Option<I::Item> {
                    (**self).next()
                }
            }
        }
        pub use self::iterator::Iterator;

        mod collect {
            pub trait IntoIterator {
                type Item;
                type IntoIter: Iterator<Item = Self::Item>;
                #[lang = "into_iter"]
                fn into_iter(self) -> Self::IntoIter;
            }
            impl<I: Iterator> IntoIterator for I {
                type Item = I::Item;
                type IntoIter = I;
                fn into_iter(self) -> I {
                    self
                }
            }
        }
        pub use self::collect::IntoIterator;
    }
    pub use self::traits::{IntoIterator, Iterator};
}
// endregion:iterator

// region:derive
mod macros {
    pub(crate) mod builtin {
        #[rustc_builtin_macro]
        pub macro derive($item:item) {
            /* compiler built-in */
        }
    }
}
// endregion:derive

// region:bool_impl
#[lang = "bool"]
impl bool {
    pub fn then<T, F: FnOnce() -> T>(self, f: F) -> Option<T> {
        if self {
            Some(f())
        } else {
            None
        }
    }
}
// endregion:bool_impl

pub mod prelude {
    pub mod v1 {
        pub use crate::{
            clone::Clone,                       // :clone
            cmp::{Eq, PartialEq},               // :eq
            cmp::{Ord, PartialOrd},             // :ord
            convert::AsRef,                     // :as_ref
            convert::{From, Into},              // :from
            default::Default,                   // :default
            iter::{IntoIterator, Iterator},     // :iterator
            macros::builtin::derive,            // :derive
            marker::Copy,                       // :copy
            marker::Sized,                      // :sized
            mem::drop,                          // :drop
            ops::Drop,                          // :drop
            ops::{Fn, FnMut, FnOnce},           // :fn
            option::Option::{self, None, Some}, // :option
            result::Result::{self, Err, Ok},    // :result
        };
    }

    pub mod rust_2015 {
        pub use super::v1::*;
    }

    pub mod rust_2018 {
        pub use super::v1::*;
    }

    pub mod rust_2021 {
        pub use super::v1::*;
    }
}

#[prelude_import]
#[allow(unused)]
use prelude::v1::*;
