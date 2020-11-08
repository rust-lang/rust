//! This crate exports a macro `enum_from_primitive!` that wraps an
//! `enum` declaration and automatically adds an implementation of
//! `num::FromPrimitive` (reexported here), to allow conversion from
//! primitive integers to the enum. It therefore provides an
//! alternative to the built-in `#[derive(FromPrimitive)]`, which
//! requires the unstable `std::num::FromPrimitive` and is disabled in
//! Rust 1.0.
//!
//! # Example
//!
//! ```
//! #[macro_use] extern crate enum_primitive;
//! extern crate num_traits;
//! use num_traits::FromPrimitive;
//!
//! enum_from_primitive! {
//! #[derive(Debug, PartialEq)]
//! enum FooBar {
//!     Foo = 17,
//!     Bar = 42,
//!     Baz,
//! }
//! }
//!
//! fn main() {
//!     assert_eq!(FooBar::from_i32(17), Some(FooBar::Foo));
//!     assert_eq!(FooBar::from_i32(42), Some(FooBar::Bar));
//!     assert_eq!(FooBar::from_i32(43), Some(FooBar::Baz));
//!     assert_eq!(FooBar::from_i32(91), None);
//! }
//! ```

pub mod num_traits {
    pub trait FromPrimitive: Sized {
        fn from_i64(n: i64) -> Option<Self>;
        fn from_u64(n: u64) -> Option<Self>;
    }
}

pub use num_traits::FromPrimitive;
pub use std::option::Option;

/// Helper macro for internal use by `enum_from_primitive!`.
#[macro_export]
macro_rules! enum_from_primitive_impl_ty {
    ($meth:ident, $ty:ty, $name:ident, $( $variant:ident )*) => {
        #[allow(non_upper_case_globals, unused)]
        fn $meth(n: $ty) -> $crate::Option<Self> {
            $( if n == $name::$variant as $ty {
                $crate::Option::Some($name::$variant)
            } else )* {
                $crate::Option::None
            }
        }
    };
}

/// Helper macro for internal use by `enum_from_primitive!`.
#[macro_export]
#[macro_use(enum_from_primitive_impl_ty)]
macro_rules! enum_from_primitive_impl {
    ($name:ident, $( $variant:ident )*) => {
        impl $crate::FromPrimitive for $name {
            enum_from_primitive_impl_ty! { from_i64, i64, $name, $( $variant )* }
            enum_from_primitive_impl_ty! { from_u64, u64, $name, $( $variant )* }
        }
    };
}

/// Wrap this macro around an `enum` declaration to get an
/// automatically generated implementation of `num::FromPrimitive`.
#[macro_export]
#[macro_use(enum_from_primitive_impl)]
macro_rules! enum_from_primitive {
    (
        $( #[$enum_attr:meta] )*
        enum $name:ident {
            $( $( #[$variant_attr:meta] )* $variant:ident ),+
            $( = $discriminator:expr, $( $( #[$variant_two_attr:meta] )* $variant_two:ident ),+ )*
        }
    ) => {
        $( #[$enum_attr] )*
        enum $name {
            $( $( #[$variant_attr] )* $variant ),+
            $( = $discriminator, $( $( #[$variant_two_attr] )* $variant_two ),+ )*
        }
        enum_from_primitive_impl! { $name, $( $variant )+ $( $( $variant_two )+ )* }
    };

    (
        $( #[$enum_attr:meta] )*
        enum $name:ident {
            $( $( $( #[$variant_attr:meta] )* $variant:ident ),+ = $discriminator:expr ),*
        }
    ) => {
        $( #[$enum_attr] )*
        enum $name {
            $( $( $( #[$variant_attr] )* $variant ),+ = $discriminator ),*
        }
        enum_from_primitive_impl! { $name, $( $( $variant )+ )* }
    };

    (
        $( #[$enum_attr:meta] )*
        enum $name:ident {
            $( $( #[$variant_attr:meta] )* $variant:ident ),+
            $( = $discriminator:expr, $( $( #[$variant_two_attr:meta] )* $variant_two:ident ),+ )*,
        }
    ) => {
        $( #[$enum_attr] )*
        enum $name {
            $( $( #[$variant_attr] )* $variant ),+
            $( = $discriminator, $( $( #[$variant_two_attr] )* $variant_two ),+ )*,
        }
        enum_from_primitive_impl! { $name, $( $variant )+ $( $( $variant_two )+ )* }
    };

    (
        $( #[$enum_attr:meta] )*
        enum $name:ident {
            $( $( $( #[$variant_attr:meta] )* $variant:ident ),+ = $discriminator:expr ),+,
        }
    ) => {
        $( #[$enum_attr] )*
        enum $name {
            $( $( $( #[$variant_attr] )* $variant ),+ = $discriminator ),+,
        }
        enum_from_primitive_impl! { $name, $( $( $variant )+ )+ }
    };

    (
        $( #[$enum_attr:meta] )*
        pub enum $name:ident {
            $( $( #[$variant_attr:meta] )* $variant:ident ),+
            $( = $discriminator:expr, $( $( #[$variant_two_attr:meta] )* $variant_two:ident ),+ )*
        }
    ) => {
        $( #[$enum_attr] )*
        pub enum $name {
            $( $( #[$variant_attr] )* $variant ),+
            $( = $discriminator, $( $( #[$variant_two_attr] )* $variant_two ),+ )*
        }
        enum_from_primitive_impl! { $name, $( $variant )+ $( $( $variant_two )+ )* }
    };

    (
        $( #[$enum_attr:meta] )*
        pub enum $name:ident {
            $( $( $( #[$variant_attr:meta] )* $variant:ident ),+ = $discriminator:expr ),*
        }
    ) => {
        $( #[$enum_attr] )*
        pub enum $name {
            $( $( $( #[$variant_attr] )* $variant ),+ = $discriminator ),*
        }
        enum_from_primitive_impl! { $name, $( $( $variant )+ )* }
    };

    (
        $( #[$enum_attr:meta] )*
        pub enum $name:ident {
            $( $( #[$variant_attr:meta] )* $variant:ident ),+
            $( = $discriminator:expr, $( $( #[$variant_two_attr:meta] )* $variant_two:ident ),+ )*,
        }
    ) => {
        $( #[$enum_attr] )*
        pub enum $name {
            $( $( #[$variant_attr] )* $variant ),+
            $( = $discriminator, $( $( #[$variant_two_attr] )* $variant_two ),+ )*,
        }
        enum_from_primitive_impl! { $name, $( $variant )+ $( $( $variant_two )+ )* }
    };

    (
        $( #[$enum_attr:meta] )*
        pub enum $name:ident {
            $( $( $( #[$variant_attr:meta] )* $variant:ident ),+ = $discriminator:expr ),+,
        }
    ) => {
        $( #[$enum_attr] )*
        pub enum $name {
            $( $( $( #[$variant_attr] )* $variant ),+ = $discriminator ),+,
        }
        enum_from_primitive_impl! { $name, $( $( $variant )+ )+ }
    };
}
