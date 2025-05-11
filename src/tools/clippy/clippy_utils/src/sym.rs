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
    abs,
    as_bytes,
    as_deref_mut,
    as_deref,
    as_mut,
    Binary,
    build_hasher,
    cargo_clippy: "cargo-clippy",
    Cargo_toml: "Cargo.toml",
    cast,
    chars,
    CLIPPY_ARGS,
    CLIPPY_CONF_DIR,
    clone_into,
    cloned,
    collect,
    contains,
    copied,
    CRLF: "\r\n",
    Current,
    ends_with,
    exp,
    extend,
    finish_non_exhaustive,
    finish,
    flat_map,
    for_each,
    from_raw,
    from_str_radix,
    get,
    insert,
    int_roundings,
    into_bytes,
    into_owned,
    IntoIter,
    is_ascii,
    is_empty,
    is_err,
    is_none,
    is_ok,
    is_some,
    last,
    LF: "\n",
    LowerExp,
    LowerHex,
    max,
    min,
    mode,
    msrv,
    Octal,
    or_default,
    parse,
    push,
    regex,
    reserve,
    resize,
    restriction,
    rustfmt_skip,
    set_len,
    set_mode,
    set_readonly,
    signum,
    split_whitespace,
    split,
    Start,
    take,
    TBD,
    then_some,
    to_digit,
    to_owned,
    unused_extern_crates,
    unwrap_err,
    unwrap_or_default,
    UpperExp,
    UpperHex,
    V4,
    V6,
    Weak,
    with_capacity,
    wrapping_offset,
}
