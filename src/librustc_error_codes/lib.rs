//! This library is used to gather all error codes into one place,
//! the goal being to make their maintenance easier.

macro_rules! register_diagnostics {
    ($($ecode:ident: $message:expr,)* ; $($code:ident,)*) => (
        pub static DIAGNOSTICS: &[(&str, &str)] = &[
            $( (stringify!($ecode), $message), )*
        ];

        $(
            pub const $ecode: () = ();
        )*
        $(
            pub const $code: () = ();
        )*
    )
}

mod error_codes;

pub use error_codes::*;
