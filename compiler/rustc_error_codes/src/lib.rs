#![deny(invalid_codeblock_attributes)]
//! This library is used to gather all error codes into one place,
//! the goal being to make their maintenance easier.

macro_rules! register_diagnostics {
    ($($ecode:ident: $message:expr,)* ; $($code:ident,)*) => (
        pub static DIAGNOSTICS: &[(&str, Option<&str>)] = &[
            $( (stringify!($ecode), Some($message)), )*
            $( (stringify!($code), None), )*
        ];
    )
}

mod error_codes;
pub use error_codes::DIAGNOSTICS;
