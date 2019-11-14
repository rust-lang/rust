//! This library is used to gather all error codes into one place. The goal
//! being to make their maintenance easier.

#[macro_export]
macro_rules! register_diagnostics {
    ($($ecode:ident: $message:expr,)*) => (
        $crate::register_diagnostics!{$($ecode:$message,)* ;}
    );

    ($($ecode:ident: $message:expr,)* ; $($code:ident,)*) => (
        pub static DIAGNOSTICS: &[(&str, &str)] = &[
            $( (stringify!($ecode), $message), )*
        ];

        $(
            pub const $ecode: &str = $message;
        )*
        $(
            pub const $code: () = ();
        )*
    )
}

mod error_codes;

pub use error_codes::*;
