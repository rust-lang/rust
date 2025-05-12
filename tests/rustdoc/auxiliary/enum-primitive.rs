// Copyright (c) 2015 Anders Kaseorg <andersk@mit.edu>

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// “Software”), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:

// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

pub use std::option::Option;
pub use num_traits::FromPrimitive;

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
