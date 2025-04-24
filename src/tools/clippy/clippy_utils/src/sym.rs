#![allow(non_upper_case_globals)]

use rustc_span::symbol::{PREDEFINED_SYMBOLS_COUNT, Symbol};

#[doc(no_inline)]
pub use rustc_span::sym::*;

macro_rules! val {
    ($name:ident) => {
        stringify!($name)
    };
    ($name:ident $value:literal) => {
        $value
    };
}

macro_rules! generate {
    ($($name:ident $(: $value:literal)? ,)*) => {
        /// To be supplied to `rustc_interface::Config`
        pub const EXTRA_SYMBOLS: &[&str] = &[
            $(
                val!($name $($value)?),
            )*
        ];

        $(
            pub const $name: Symbol = Symbol::new(PREDEFINED_SYMBOLS_COUNT + ${index()});
        )*
    };
}

generate! {
    as_bytes,
    as_deref_mut,
    as_deref,
    as_mut,
    Binary,
    Cargo_toml: "Cargo.toml",
    CLIPPY_ARGS,
    CLIPPY_CONF_DIR,
    cloned,
    contains,
    copied,
    Current,
    get,
    insert,
    int_roundings,
    IntoIter,
    is_empty,
    is_ok,
    is_some,
    LowerExp,
    LowerHex,
    msrv,
    Octal,
    or_default,
    regex,
    rustfmt_skip,
    Start,
    to_owned,
    unused_extern_crates,
    unwrap_err,
    unwrap_or_default,
    UpperExp,
    UpperHex,
    V4,
    V6,
    Weak,
}
