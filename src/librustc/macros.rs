// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! enum_from_u32 {
    ($(#[$attr:meta])* pub enum $name:ident {
        $($variant:ident = $e:expr,)*
    }) => {
        $(#[$attr])*
        pub enum $name {
            $($variant = $e),*
        }

        impl $name {
            pub fn from_u32(u: u32) -> Option<$name> {
                $(if u == $name::$variant as u32 {
                    return Some($name::$variant)
                })*
                None
            }
        }
    };
    ($(#[$attr:meta])* pub enum $name:ident {
        $($variant:ident,)*
    }) => {
        $(#[$attr])*
        pub enum $name {
            $($variant,)*
        }

        impl $name {
            pub fn from_u32(u: u32) -> Option<$name> {
                $(if u == $name::$variant as u32 {
                    return Some($name::$variant)
                })*
                None
            }
        }
    }
}

#[macro_export]
macro_rules! bug {
    () => ( bug!("impossible case reached") );
    ($($message:tt)*) => ({
        $crate::session::bug_fmt(file!(), line!(), format_args!($($message)*))
    })
}

#[macro_export]
macro_rules! span_bug {
    ($span:expr, $($message:tt)*) => ({
        $crate::session::span_bug_fmt(file!(), line!(), $span, format_args!($($message)*))
    })
}

#[macro_export]
macro_rules! type_err {
    ($infcx:expr, $origin: expr, $values: expr, $terr: expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        $infcx.report_and_explain_type_error_with_code(
            $origin,
            $values,
            &$terr,
            &format!($($message)*),
            stringify!($code))
    })
}
