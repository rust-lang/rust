#![unstable(feature = "non_portable_conversion", issue = /* FIXME */ "none")]

//! Target-specific functionality
//!
//! ## Background: `From` and `Into` are portable
//!
//! The `From` and `Into` traits are in the prelude, so they don’t need to be imported with `use`.
//! They provide conversions that are infallible.
//! For example, `From<u32> for u64` is implemented:
//!
//! ```
//! assert_eq!(u64::from(7_u32), 7_u64);
//! ```
//!
//! … but `From<u64> for u32` is not, because larger values cannot be represented:
//!
//! ```compile_fail,E0277
//! let x = 7_u64;
//! // error: `std::convert::From<u32>` is not implemented for `u16`
//! let _ = u32::from(x); // (What if `x` was `7_000_000_000_000_u64` ?)
//! ```
//!
//! Additionally, `From` and `Into` impls are portable:
//! they only exist when they are infallible regardless of the target.
//! For example, converting `u64` to `usize` would be infallible
//! if the target happens to have 64-bit pointers,
//! but the `From` trait still doesn’t allow it:
//!
//! ```compile_fail,E0277
//! let x = 7_u64;
//! // error: `std::convert::From<u64>` is not implemented for `usize`
//! let _ = usize::from(x);
//! ```
//!
//! This conversion is possible with the `TryFrom` trait:
//!
//! ```
//! use std::convert::TryFrom;
//!
//! assert_eq!(usize::try_from(7_u64).unwrap(), 7_usize);
//! ```
//!
//! However, because `try_from` is fallible, this may require using `.unwrap()`.
//! In a less trivial case, it may not be obvious to readers of the code that
//! this can never panic (on 64-bit platforms).
//!
//! ## Non-portable conversion traits
//!
//! This module provides integer conversion traits that are only available for some targets.
//! They provide the "missing" integer conversions
//! for code that only cares about portability to some platforms.
//! For example, an application that only runs on 64-bit servers could use:
//!
//! ```
//! #![feature(non_portable_conversion)]
//!
//! # // Make this test a no-op on non-64-bit platforms
//! # #[cfg(target_pointer_width = "64")]
//! use std::target::PointerWidthGe64From;
//!
//! # #[cfg(target_pointer_width = "64")]
//! assert_eq!(usize::target_from(7_u64), 7_usize);
//! ```
//!
//! Here the code does not have a panic branch at all.
//! In return, it does not compile on some targets.
//!
//! ```compile_fail,E0432
//! #![feature(non_portable_conversion)]
//!
//! // These two never exist at the same time:
//! use std::target::PointerWidthGe64From;
//! use std::target::PointerWidthLe32From;
//! // error[E0432]: unresolved import
//! ```
//!
//! The mandatory import `std::target` is an indication to writers and readers
//! of non-portable, target-specific code.
//! This is similar to the `std::os` module.
//!
//! The trait names contain `Ge` or `Le` which mean “greater then or equal”
//! and “less than or equal” respectively,
//! like in the `ge` or `le` methods of the `PartialOrd` trait.

macro_rules! common_attribute {
    ( #[$attr: meta] $( $item: item )+ ) => {
        $(
            #[$attr]
            $item
        )+
    }
}

macro_rules! target_category {
    (
        [$( $ptr_width: expr ),+]
        $doc: tt
        $FromTrait: ident $( : $BlanketImplForOtherFromTrait: ident )?
        $IntoTrait: ident
        $( $from_ty: ty => $to_ty: ty, )+
    ) => {
        common_attribute! {
            #[cfg(any(doc, $(target_pointer_width = $ptr_width),+ ))]

            /// Similar to `convert::From`, only available for targets where the pointer size is
            #[doc = $doc]
            /// bits.
            pub trait $FromTrait<T>: Sized {
                /// Performs the conversion.
                fn target_from(_: T) -> Self;
            }

            /// Similar to `convert::Into`, only available for targets where the pointer size is
            #[doc = $doc]
            /// bits.
            pub trait $IntoTrait<T>: Sized {
                /// Performs the conversion.
                fn target_into(self) -> T;
            }

            // From implies Into
            impl<T, U> $IntoTrait<U> for T
            where
                U: $FromTrait<T>,
            {
                fn target_into(self) -> U {
                    U::target_from(self)
                }
            }

            $(
                // For example: if the pointer width >= 64 bits, then it is also >= 32 bits
                impl<T, U> $FromTrait<U> for T
                where
                    T: $BlanketImplForOtherFromTrait<U>,
                {
                    fn target_from(x: U) -> T {
                        $BlanketImplForOtherFromTrait::target_from(x)
                    }
                }
            )?

            $(
                impl $FromTrait<$from_ty> for $to_ty {
                    fn target_from(x: $from_ty) -> $to_ty { x as $to_ty }
                }
            )+
        }
    }
}

target_category! {
    ["64" /*, "128", ... */]
    "greater than or equal (`Ge`) to 64"
    PointerWidthGe64From: PointerWidthGe32From
    PointerWidthGe64Into
    u32 => isize,
    u64 => usize,
    i64 => isize,
}

target_category! {
    ["32", "64" /*, "128", ... */]
    "greater than or equal (`Ge`) to 32"
    PointerWidthGe32From
    PointerWidthGe32Into
    u16 => isize,
    u32 => usize,
    i32 => isize,
}

target_category! {
    ["16"]
    "less than or equal (`Le`) to 16"
    PointerWidthLe16From: PointerWidthLe32From
    PointerWidthLe16Into
    usize => u16,
    isize => i16,
    usize => i32,
}

target_category! {
    ["16", "32"]
    "less than or equal (`Le`) to 32"
    PointerWidthLe32From: PointerWidthLe64From
    PointerWidthLe32Into
    usize => u32,
    isize => i32,
    usize => i64,
}

target_category! {
    ["16", "32", "64"]
    "less than or equal (`Le`) to 64"
    PointerWidthLe64From /* : PointerWidthLe128From */
    PointerWidthLe64Into
    usize => u64,
    isize => i64,
    usize => i128,

    // If adding `PointerWidthLe128From`, these should move there:
    usize => u128,
    isize => i128,
}
